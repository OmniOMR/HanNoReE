"""
Inference with a trained model. A trained model is a training checkpoint (see train.py): it holds the weights, the
token vocabulary and the model configuration, so loading it needs neither the training data nor its arguments, and
nothing is downloaded (the MuSViT backbone is built from the stored ViT configuration). It also holds the optimizer
state, so the same file can be trained further (`hannoree train --resume` or `--init-from`).

    hannoree transcribe --model hannoree.pt staves/*.png --output-dir transcriptions
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .data import TokenVocabulary, decode_stave_band, pad_stave_bands
from .model import HanNoReEModel, model_arguments


def load_checkpoint(path: str | Path) -> dict:
    """A training checkpoint. weights_only=False: it also holds optimizer and RNG state (numpy arrays, tuples)."""
    return torch.load(path, map_location="cpu", weights_only=False)


class Transcriber:
    """A trained model loaded for inference: stave images in, LMX out. Made by load_model."""

    def __init__(self, checkpoint: dict, device: torch.device, bf16: bool = False):
        config = checkpoint["config"]
        self.input = {k: config["input"][k] for k in ("stave_height", "stave_width", "max_stave_width")}
        self.vocab = TokenVocabulary.from_dict(checkpoint["vocab"])
        self.epoch = checkpoint.get("epoch")
        self.device = device

        model = HanNoReEModel(**model_arguments(config["model"]), musvit_bf16=bf16)
        model.load_state_dict(checkpoint["model"])
        self.model = model.to(device).eval()

    def prepare(self, image_bytes: bytes) -> torch.Tensor:
        """An encoded stave image (PNG, JPEG, ...) -> the model's input, preprocessed exactly as in training."""
        return decode_stave_band(
            image_bytes, self.input["stave_height"], self.input["stave_width"], self.input["max_stave_width"]
        )

    def predict(self, images: list[torch.Tensor]) -> list[str]:
        """
        LMX transcriptions of prepared stave images, in one batch padded with white as in training. A batch that runs
        out of GPU memory is split in halves, so that only a single stave that does not fit fails.
        """
        try:
            batch, widths = pad_stave_bands(images)
            generated = self.model.generate(batch.to(self.device), widths)
            return [self.vocab.decode_prediction(ids) for ids in generated.cpu().tolist()]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(images) == 1:
                raise
            print(f"Out of GPU memory with {len(images)} staves; splitting the batch.", file=sys.stderr)
            half = len(images) // 2
            return self.predict(images[:half]) + self.predict(images[half:])

    def transcribe(self, images: list[bytes | str | Path], batch_size: int = 16) -> list[str]:
        """LMX transcriptions of stave images, given as encoded bytes or file paths."""
        transcriptions: list[str] = []
        for start in range(0, len(images), batch_size):
            chunk = [image if isinstance(image, bytes) else Path(image).read_bytes() for image in images[start:start + batch_size]]
            transcriptions += self.predict([self.prepare(image) for image in chunk])
        return transcriptions


def load_model(path: str | Path, device: str | torch.device | None = None, bf16: bool = False) -> Transcriber:
    """
    Loads a trained model (a training checkpoint, .pt) for inference. `device` defaults to CUDA if available;
    `bf16` runs the MuSViT backbone in bfloat16 autocast (Ampere+ GPUs).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return Transcriber(load_checkpoint(path), device, bf16)


def write_transcription(lmx: str, output_dir: Path, stem: str) -> str | None:
    """
    Writes `<stem>.lmx` and the MusicXML decoded from it, `<stem>.musicxml`, to `output_dir`. Returns the error if
    the LMX cannot be decoded into MusicXML (then only the LMX is written), else None.

    The LMX decoder fails on tokens before the first measure, so they are dropped from the MusicXML (not from the LMX).
    """
    from zeus.musicxml.lmx_to_musicxml import LmxDecodingError, lmx_to_musicxml

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.lmx").write_text(lmx + "\n", encoding="utf-8")
    tokens = lmx.split()
    first_measure = tokens.index("measure") if "measure" in tokens else 0
    try:
        musicxml = lmx_to_musicxml(" ".join(tokens[first_measure:]))
    except LmxDecodingError as e:
        (output_dir / f"{stem}.musicxml").unlink(missing_ok=True)
        return str(e)
    (output_dir / f"{stem}.musicxml").write_text(musicxml, encoding="utf-8")
    return None


def define_parser(parser):
    parser.add_argument("images", type=Path, nargs="+", help="Images of single staves (PNG, JPEG, ...).")
    parser.add_argument("--model", type=Path, required=True, help="The trained model (a training checkpoint, .pt).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write <image name>.lmx and <image name>.musicxml (default: beside each image).")
    parser.add_argument("--batch-size", type=int, default=16, help="Staves per forward pass.")
    parser.add_argument("--device", help="'cuda' or 'cpu' (default: cuda if available).")
    parser.add_argument("--bf16", action="store_true", help="Run the MuSViT backbone in bfloat16 autocast (Ampere+ GPUs).")


def run(args):
    transcriber = load_model(args.model, args.device, args.bf16)
    print(f"Transcribing {len(args.images):,} staves with '{args.model}' (epoch {transcriber.epoch}) on {transcriber.device}.")
    failures = 0
    transcriptions = transcriber.transcribe(args.images, args.batch_size)
    for image, lmx in zip(args.images, transcriptions, strict=True):
        error = write_transcription(lmx, args.output_dir or image.parent, image.stem)
        if error is not None:
            failures += 1
            print(f"{image}: only LMX written, {error}", file=sys.stderr)
    print(f"Done: {len(args.images) - failures:,} with MusicXML, {failures:,} with LMX only.")
