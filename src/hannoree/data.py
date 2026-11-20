"""
Data for the HanNoReE model:
- Image preprocessing: an encoded stave image -> the RGB stave band the MuSViT backbone reads, shared by training
  and inference, so that both see identical inputs.
- The token vocabulary (LMX token <-> ID).
- Loading the pickled splits of Zeus datasets: the datasets are subfolders of a common directory
  (e.g. datasets/omniomr, datasets/dolores), each with its own pickled splits (samples.{train,validation,test}.pickle),
  made with `zeus musicorpus` and `zeus pickle` of the Zeus package.
- The dataset, its collate function and the batch sampler used in training and evaluation.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import to_tensor
from zeus.data.zeus_dataset import ZeusDataset, ZeusDatasetSample

from .augmentation import ZeusAugmentation

MUSVIT_PATCH_SIZE = 16


# ==============================================================================
# Image Preprocessing
# ==============================================================================

def aspect_stave_width(image_size: tuple[int, int], height: int, max_width: int, patch_size: int = MUSVIT_PATCH_SIZE) -> int:
    """
    Width of a stave of `image_size` (width, height) resized to `height` with its aspect ratio kept, rounded to
    a multiple of the patch size and capped at `max_width` (longer staves are squashed).
    """
    w, h = image_size
    columns = max(1, round(height * w / h / patch_size))
    return min(columns * patch_size, max_width)


def decode_stave_band(image_bytes: bytes, height: int = 96, width: int | None = 1024, max_width: int = 2048) -> torch.Tensor:
    """
    Encoded stave image -> (3, height, width) RGB stave band in [0, 1], without normalization (as the MuSViT
    documentation prescribes). With width=None, the band keeps the stave's aspect ratio (see aspect_stave_width).
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        band_width = width or aspect_stave_width(image.size, height, max_width)
        return to_tensor(image.convert("RGB").resize((band_width, height)))
    except OSError as e:
        raise ValueError("The image could not be decoded.") from e


# ==============================================================================
# Token Vocabulary
# ==============================================================================

class TokenVocabulary:
    """Manages string token to integer ID mapping, mirroring Zeus TokenMap."""
    def __init__(self, special_tokens: list[str] | None = None):
        self.special_tokens = special_tokens or ["<bos>", "<eos>", "<pad>", "<unk>"]
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

        for tok in self.special_tokens:
            self.add_token(tok)

    @property
    def bos_idx(self) -> int:
        return self.token2id[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self.token2id[self.eos_token]

    @property
    def pad_idx(self) -> int:
        return self.token2id[self.pad_token]

    @property
    def unk_idx(self) -> int:
        return self.token2id[self.unk_token]

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            return idx
        return self.token2id[token]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token2id.get(t, self.unk_idx) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id2token.get(i, self.unk_token) for i in ids]

    def decode_prediction(self, ids: list[int]) -> str:
        """A generated token sequence as an LMX string: cut at the first EOS, without padding and BOS."""
        clean = []
        for t in ids:
            if t == self.eos_idx:
                break
            if t not in (self.pad_idx, self.bos_idx):
                clean.append(t)
        return " ".join(self.decode(clean))

    def __len__(self) -> int:
        return len(self.token2id)

    def to_dict(self) -> dict:
        return {"token2id": self.token2id, "special_tokens": self.special_tokens}

    @classmethod
    def from_dict(cls, data: dict) -> TokenVocabulary:
        # Older checkpoints stored the bare token2id mapping
        if "token2id" not in data:
            data = {"token2id": data}
        vocab = cls(special_tokens=data.get("special_tokens"))
        vocab.token2id = dict(data["token2id"])
        vocab.id2token = {int(v): k for k, v in vocab.token2id.items()}
        return vocab

    def save(self, filepath: str | Path):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> TokenVocabulary:
        with open(filepath, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def build_from_samples(cls, samples: list[ZeusDatasetSample]) -> TokenVocabulary:
        """Constructs vocabulary from all LMX tokens present in training samples."""
        vocab = cls()
        for sample in samples:
            for token in sample.lmx.split():
                vocab.add_token(token)
        return vocab


# ==============================================================================
# Dataset Selection and Loading of Pickled Splits
# ==============================================================================

SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "dev": ("dev", "val", "validation"),
    "test": ("test",),
}

SplitSamples = dict[str, list[ZeusDatasetSample]]
"""Samples of one split, keyed by the name of the dataset they come from."""


def discover_datasets(dataset_dir: str | Path) -> list[str]:
    """Names of all subfolders of `dataset_dir` that contain a pickled Zeus split."""
    base = Path(dataset_dir)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and any(p.glob("samples.*.pickle")))


def resolve_dataset_folder(spec: str | Path, dataset_dir: str | Path) -> Path:
    """Resolves a dataset name (subfolder of `dataset_dir`) or a path to a dataset folder."""
    folder = Path(dataset_dir) / spec
    if folder.is_dir():
        return folder
    if Path(spec).is_dir():
        return Path(spec)
    available = ", ".join(discover_datasets(dataset_dir)) or "none"
    raise FileNotFoundError(f"Dataset '{spec}' not found. Datasets available in '{dataset_dir}': {available}")


def resolve_split_pickle(spec: str | Path, dataset_dir: str | Path, split: str) -> Path:
    """
    Resolves a dataset specification to the pickle of its predefined `split`.
    `spec` is a dataset name, a path to a dataset folder, or a path to a .pickle file (used as-is).
    """
    if Path(spec).suffix == ".pickle":
        if not Path(spec).is_file():
            raise FileNotFoundError(f"Zeus dataset pickle not found at: '{spec}'")
        return Path(spec)

    folder = resolve_dataset_folder(spec, dataset_dir)
    for alias in SPLIT_ALIASES[split]:
        candidate = folder / f"samples.{alias}.pickle"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Dataset '{spec}' has no pickled '{split}' split in '{folder}'. "
        f"Pickle it with: zeus pickle {folder}/samples.<split>.txt"
    )


def load_pickle(pickle_path: Path) -> list[ZeusDatasetSample]:
    return ZeusDataset.load_from_pickle_file(pickle_path).samples


def load_split(specs: list[str], dataset_dir: str | Path, split: str) -> SplitSamples:
    """Loads the predefined `split` of each selected dataset. A dataset is named after its folder."""
    loaded: SplitSamples = {}
    for spec in specs:
        pickle_path = resolve_split_pickle(spec, dataset_dir, split)
        name = pickle_path.parent.name
        if name in loaded:
            raise ValueError(f"Dataset '{name}' is selected more than once for the '{split}' split.")
        loaded[name] = load_pickle(pickle_path)
        print(f"Loaded {len(loaded[name]):,} {split} samples of '{name}' from '{pickle_path}'")
    return loaded


# ==============================================================================
# Stave Dataset
# ==============================================================================

class StaveOMRDataset(Dataset):
    """
    Dataset backed by in-memory ZeusDatasetSample objects. An item is the (3, stave_height, stave_width) RGB stave
    band of a sample's image (see decode_stave_band) and its LMX token IDs. With stave_width=None, the band keeps the
    stave's aspect ratio, its width rounded to a multiple of the patch size and capped at max_stave_width.

    With `augment` (a Zeus augmentation pipeline, see augmentation.py), stave bands are augmented when an item is
    requested as (index, seed), as the training sampler does (StaveBatchSampler with sample_seeds); a plain
    index gives the unaugmented sample. The seed makes the augmentation independent of the DataLoader workers, so
    that resumed training sees the same augmentations as an uninterrupted run.
    """
    def __init__(
        self,
        samples: SplitSamples,
        vocab: TokenVocabulary,
        stave_height: int = 96,
        stave_width: int | None = 1024,
        max_stave_width: int = 2048,
        augment: str = "",  # Zeus augmentation pipeline, e.g. 'h:8,rotate:1,v:4'
    ):
        self.items = [(name, sample) for name, dataset_samples in samples.items() for sample in dataset_samples]
        self.vocab = vocab
        self.stave_height = stave_height
        self.stave_width = stave_width
        self.max_stave_width = max_stave_width
        # The horizontal shift keeps the width of stave bands: fixed, or a multiple of the patch size up to the cap
        self.augmentation = ZeusAugmentation(augment, keep_width=True) if augment else None

    def __len__(self) -> int:
        return len(self.items)

    def gold_lmx(self) -> list[str]:
        """Gold LMX strings in dataset order."""
        return [sample.lmx for _, sample in self.items]

    def dataset_names(self) -> list[str]:
        """Name of the source dataset of each sample, in dataset order."""
        return [name for name, _ in self.items]

    def target_lengths(self) -> list[int]:
        """Number of LMX tokens of each sample, in dataset order."""
        return [len(sample.lmx.split()) for _, sample in self.items]

    def __getitem__(self, item: int | tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """`item` is a sample index, or an (index, seed) pair to augment the sample with that seed."""
        idx, seed = item if isinstance(item, tuple) else (item, None)
        _, sample = self.items[idx]
        token_tensor = torch.tensor(self.vocab.encode(sample.lmx.split()), dtype=torch.long)

        try:
            image = decode_stave_band(sample.image, self.stave_height, self.stave_width, self.max_stave_width)
        except ValueError as e:
            raise ValueError(f"Sample '{sample.sample_name}': {e}") from e

        # Last step before the model, as in Zeus
        if self.augmentation is not None and seed is not None:
            image = self.augmentation(image, torch.Generator().manual_seed(seed))
        return image, token_tensor


# ==============================================================================
# Collate Function
# ==============================================================================

def pad_stave_bands(bands: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(3, H, W_i) stave bands -> the (B, 3, H, W_max) batch padded with white, and the (B,) unpadded widths."""
    widths = torch.tensor([band.shape[-1] for band in bands], dtype=torch.long)
    max_width = int(widths.max())
    batch = torch.stack([torch.nn.functional.pad(band, (0, max_width - band.shape[-1]), value=1.0) for band in bands])
    return batch, widths


class StaveCollate:
    """Collates variable-width stave bands and variable-length token targets."""
    def __init__(self, pad_idx: int = 2, bos_idx: int = 0, eos_idx: int = 1):
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __call__(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs: (B, 3, H, W_max) stave bands padded with white
            input_lengths: (B,) unpadded widths
            input_seqs: (B, L+1) teacher forcing inputs [BOS, t_1, ..., t_L]
            target_labels: (B, L+1) labels for loss [t_1, ..., t_L, EOS]
        """
        bands, targets = zip(*batch, strict=True)
        batch_inputs, input_lengths = pad_stave_bands(list(bands))

        max_len = max(len(t) for t in targets)
        batch_input_seqs = []
        batch_target_labels = []
        for t in targets:
            inp = torch.cat([torch.tensor([self.bos_idx]), t])
            lbl = torch.cat([t, torch.tensor([self.eos_idx])])

            pad_len = (max_len + 1) - len(inp)
            if pad_len > 0:
                inp = torch.nn.functional.pad(inp, (0, pad_len), value=self.pad_idx)
                lbl = torch.nn.functional.pad(lbl, (0, pad_len), value=self.pad_idx)

            batch_input_seqs.append(inp)
            batch_target_labels.append(lbl)

        return batch_inputs, input_lengths, torch.stack(batch_input_seqs), torch.stack(batch_target_labels)


# ==============================================================================
# Batching
# ==============================================================================

class StaveBatchSampler(Sampler[list[int]]):
    """
    - shuffle=True (training): random batches, seeded from the global RNG like torch's RandomSampler, so that
      seeding and resuming stay reproducible.
    - shuffle=False (evaluation): batches in dataset order.

    With `sample_seeds` (shuffle only), every sample comes as an (index, seed) pair, the seed for augmenting it. The
    seeds come from the sampler rather than from the DataLoader workers, so that a resumed run sees the same
    augmentations as an uninterrupted one, for any number of workers.
    """
    def __init__(self, num_samples: int, batch_size: int, shuffle: bool, sample_seeds: bool = False):
        if sample_seeds and not shuffle:
            raise ValueError("Sample seeds are only drawn for shuffled (training) batches.")
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_seeds = sample_seeds

    def __iter__(self):
        if not self.shuffle:
            for start in range(0, self.num_samples, self.batch_size):
                yield list(range(start, min(start + self.batch_size, self.num_samples)))
            return

        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        permutation = torch.randperm(self.num_samples, generator=generator).tolist()
        for start in range(0, self.num_samples, self.batch_size):
            batch = permutation[start:start + self.batch_size]
            if self.sample_seeds:
                seeds = torch.randint(2**62, (len(batch),), generator=generator).tolist()
                batch = list(zip(batch, seeds, strict=True))
            yield batch

    def __len__(self) -> int:
        return math.ceil(self.num_samples / self.batch_size)
