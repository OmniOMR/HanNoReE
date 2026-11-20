# HanNoReE

HanNoReE is a staff-level optical music recognition (OMR) model. It reads the image of a single stave and transcribes it into [LMX](https://github.com/OMR-Research/lmx) (linearized MusicXML), which is decoded into MusicXML.

---

## Installation

Requirements: Python 3.10 or newer. A CUDA GPU is needed for training and recommended for inference.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -e .
pip install --no-deps --ignore-requires-python -r requirements-zeus.txt
```

This installs the `hannoree` package and command. The second line installs the [Zeus](https://github.com/OmniOMR/zeus) package, pinned to a commit. Zeus provides the dataset tooling (`zeus musicorpus`, `zeus pickle`), the SER metric and the LMX-to-MusicXML decoding. It is installed without its dependencies: it pins TensorFlow 2.12, and so Python 3.10, but none of the Zeus modules used here imports TensorFlow. The dependencies they do need are listed in `pyproject.toml`.

To use the model without a clone of the repository, install from git instead of `-e .`:

```bash
pip install "hannoree @ git+https://github.com/OmniOMR/HanNoReE.git@v1.0.0"   # a release tag, branch or commit
```

The package version is derived from the git tags, e.g. `1.0.0` on the tag `v1.0.0`. An installation therefore needs the git history. A source archive without `.git`, such as GitHub's "Download ZIP", fails to build with `unable to detect version`. In that case set `SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0`.

---

## Using the released model

The trained model is attached to every [GitHub release](https://github.com/OmniOMR/HanNoReE/releases) of this repository as `hannoree.pt`. It is the training checkpoint of the selected epoch: the weights (the fine-tuned MuSViT backbone included), the token vocabulary and the model configuration, plus the optimizer state and the training history, so it can also be [trained further](#training-further). Loading it downloads nothing else and needs no Hugging Face access.

### Download

Download `hannoree.pt` from the release page, or with the [GitHub CLI](https://cli.github.com/):

```bash
gh release download --repo OmniOMR/HanNoReE --pattern hannoree.pt             # the latest release
gh release download v1.0.0 --repo OmniOMR/HanNoReE --pattern hannoree.pt      # a given release
```

The checkpoint is a pickle, loaded with `torch.load(..., weights_only=False)`, which can run code. Only load checkpoints from sources you trust.

### Transcribe stave images

```bash
hannoree transcribe --model hannoree.pt staves/*.png --output-dir transcriptions
```

For each image, this writes `<image name>.lmx` and the MusicXML decoded from it, `<image name>.musicxml` (without `--output-dir`, beside the image). Each image must be a crop of a single stave. A whole page or a grand staff would be transcribed as if it were one stave. Predicted LMX that the decoder cannot turn into MusicXML keeps its `.lmx`, and the failure is reported.

### From Python

```python
from hannoree import load_model
from zeus.musicxml.lmx_to_musicxml import LmxDecodingError, lmx_to_musicxml

model = load_model("hannoree.pt", device="cuda")      # device defaults to CUDA if available

lmx = model.transcribe(["stave1.png", "stave2.png"], batch_size=16)   # file paths or encoded image bytes
try:
    musicxml = lmx_to_musicxml(lmx[0])
except LmxDecodingError as e:
    print("Not decodable:", e)
```

`bf16=True` runs the backbone in bfloat16 on Ampere and newer GPUs, which is faster and slightly less precise. `model.prepare(image_bytes)` gives the preprocessed input tensor, and `model.predict(tensors)` decodes a batch of them. Both are used by `transcribe`.

---

## Model architecture

The model is a pre-trained [MuSViT](https://huggingface.co/PRAIG/musvit) vision transformer, fine-tuned end to end with the attention-based LSTM decoder of [Zeus](https://github.com/OmniOMR/zeus):

```text
stave image ──resize──▶ 96 x 1024 RGB band
            ──MuSViT (ViT-Base, 16 px patches)──▶ 6 x 64 patch grid of 768-d features
            ──one timestep per patch column (6 x 768 concatenated)──▶ 64 x 4608
            ──adapter: Linear + LayerNorm + GELU, BiLSTM (sum), residual LayerNorm──▶ 64 x 256
            ──Zeus decoder: LSTM (256) with Bahdanau attention, greedy decoding──▶ LMX tokens
```

- **Encoder.** The MuSViT backbone reads the stave band alone, with its pre-trained position embeddings interpolated to the band's size, as the MuSViT model card recommends for fine-tuning. Each of the 64 patch columns becomes one timestep. Its 6 patch rows are concatenated, so the vertical (pitch) position of every patch is kept. A small adapter projects the timesteps to 256 dimensions and contextualizes them with a bidirectional LSTM.
- **Decoder.** It is the Zeus decoder: a single-layer LSTM with additive (Bahdanau) attention over the encoder output. It generates LMX tokens greedily, up to a length fixed at training time.
- **Training.** The backbone is frozen at first, while the randomly initialized adapter and decoder learn to read its features. It is then fine-tuned with a much lower learning rate (see [Training](#training)).

---

## Datasets

The model is trained and evaluated on [MusiCorpus](https://github.com/OmniOMR/musicorpus) datasets: UFAL.OmniOMR, CVC.Dolores and the synthetic and scanned [OLiMPiC](https://github.com/ufal/olimpic-icdar24). OLiMPiC is published in its own format; [Exporting OLiMPiC](https://github.com/OmniOMR/musicorpus/blob/main/docs/exporting-olimpic-dataset.md) describes how to make the MusiCorpus datasets UFAL.OlimpicSynthetic and UFAL.OlimpicScanned from it. The datasets are converted to the Zeus dataset format and pickled with the Zeus package. Each becomes a subfolder of `datasets/` with one pickle per split:

```bash
# Convert the staves of a MusiCorpus dataset (writes samples.<split>.txt for each split of its splits.json)
zeus musicorpus --input path/to/UFAL.OmniOMR            --output datasets/omniomr      --take-staves
zeus musicorpus --input path/to/CVC.Dolores             --output datasets/dolores      --take-staves
zeus musicorpus --input path/to/UFAL.OlimpicSynthetic   --output datasets/olimpicsynth --take-staves
zeus musicorpus --input path/to/UFAL.OlimpicScanned     --output datasets/olimpicscan  --take-staves

# Pickle each split into samples.<split>.pickle beside the .txt
for dataset in omniomr dolores olimpicsynth; do
    for split in train validation test; do zeus pickle datasets/$dataset/samples.$split.txt; done
done
# The scanned OLiMPiC has no training data, so its train split is empty and is left out
for split in validation test; do zeus pickle datasets/olimpicscan/samples.$split.txt; done
```

- The conversion uses POSIX paths, so it runs on Linux or macOS, not on Windows. Staves whose MusicXML cannot be encoded as LMX are logged and skipped.
- The conversion replaces invisible header clefs with a G clef and invisible key signatures with C major, since a stave crop does not show them. The model learns these normalized transcriptions.
- The splits come from the dataset's `splits.json`. OmniOMR and Dolores call the dev split `validation`, and `dev`, `val` and `validation` are all accepted.
- Only folders with pickled splits count as datasets, so the raw MusiCorpus folders may stay in `datasets/` as well.

---

## Training

```bash
export HF_TOKEN=...     # MuSViT is gated: accept its terms on https://huggingface.co/PRAIG/musvit first
hannoree train --train dolores olimpicsynth omniomr --dev omniomr --test dolores olimpicscan olimpicsynth omniomr \
    --musvit-bf16 --output-dir experiment_results
```

The defaults of `hannoree train` are the recipe of the released model. The command above is what [`scripts/train.slurm`](scripts/train.slurm) runs. That script is a SLURM job and also runs directly with `bash`.

### The recipe

| Setting | Value | Flag |
| :--- | :--- | :--- |
| Backbone | `PRAIG/musvit` | `--musvit-model` |
| Input | staves resized to 96 x 1024 px, ignoring the aspect ratio | `--stave-height 96 --stave-width 1024` |
| Model size | 256-dimensional adapter and decoder, dropout 0.2 | `--dim 256 --dropout 0.2` |
| Optimizer | Adam, learning rate 1e-3, cosine decay to 1% over all epochs, batch size 32, gradient clipping at 1.0 | `--lr`, `--lr-decay`, `--batch-size` |
| Backbone fine-tuning | frozen for epochs 1-19, then all its weights are fine-tuned from epoch 20 with a peak learning rate of 1e-5, warming up linearly over 5 epochs | `--finetune-from 20 --finetune-lr 1e-5 --finetune-warmup-epochs 5` |
| Loss | cross-entropy with label smoothing 0.1 (validation loss without it) | `--label-smoothing 0.1` |
| Augmentation | Zeus pipeline: horizontal shift up to 8 px, rotation up to 1°, vertical shift up to 4 px | `--augment "h:8,rotate:1,v:4"` |
| Length | 300 epochs, validation SER every epoch selects the checkpoint | `--epochs 300 --evaluation-each 1` |
| Seed | 42 | `--seed 42` |

The backbone's learning rate follows the same cosine schedule as the rest of the model, scaled to `--finetune-lr`. It is 0 while the backbone is frozen. The warmup after unfreezing is updated at every step, so the first updates of the pre-trained weights stay small while Adam's moment estimates settle. `--finetune-from none` keeps the backbone frozen throughout.

`hannoree train --help` lists all options. The less obvious ones:

| Flag | Meaning |
| :--- | :--- |
| `--musvit-bf16` | Runs the backbone in bfloat16 autocast: much faster and lighter on Ampere and newer GPUs. |
| `--grad-checkpointing` | Recomputes backbone activations in the backward pass, which saves GPU memory at ~30% more compute. |
| `--stave-width aspect` | Keeps each stave's aspect ratio instead of a fixed width. The width is rounded to a multiple of 16 and capped at `--max-stave-width` (default 2048). Batches are padded with white, and the padding is masked out of the backbone's attention, the adapter and the decoder. The position grid is stretched over `--max-stave-width`, so a patch column always gets the same embedding. Needs transformers 5. |
| `--max-gen-length N` | Longest decoded sequence. The default is 1.2 times the longest training transcription. A warning is printed if dev or test transcriptions are longer. |
| `--no-tf32` | Computes matrix multiplications in full FP32 instead of TF32 on Ampere and newer GPUs. |
| `--hf-token` | Token for the gated MuSViT models. The default is `HF_TOKEN`. |

### Selecting datasets

| Flag | Meaning |
| :--- | :--- |
| `--dataset-dir DIR` | Directory with the datasets as subfolders (default: `datasets`). |
| `--datasets A B ...` | Datasets used for all splits unless overridden (default: all datasets in `--dataset-dir`). |
| `--train A B ...` | Datasets whose `train` split is used for training (default: `--datasets`). |
| `--dev A B ...` | Datasets whose `dev` split selects the checkpoint (default: `--datasets`, or the `--train` datasets if `--datasets` is not given). |
| `--test A B ...` | Datasets whose `test` split is used for testing (same default as `--dev`). `--test` without values skips testing. |

A dataset is the name of a subfolder of `--dataset-dir`, a path to a dataset folder, or a path to a `.pickle` file. The vocabulary is built from the combined training split. The test SER is reported for the combination of the test datasets and for each one separately.

### Outputs, checkpoints and resuming

```text
experiment_results/
  best.pt, latest.pt        checkpoints: lowest validation SER, and the latest epoch
  vocab.json                the token vocabulary
  results.json              metrics of the selected checkpoint, the dataset selection and the settings
  results_table.md, .csv    the same as a one-row table
```

- A checkpoint holds the model, the optimizer and RNG state, the vocabulary, the model configuration and the training history. It is about 1.1 GB, since the fine-tuned backbone and its Adam state are included. `--save-epoch-checkpoints` also keeps one per evaluated epoch in `checkpoints/`.
- `--resume auto` continues from `latest.pt` exactly where training stopped. The losses are the same as in an uninterrupted run, for any `--num-workers`, augmentation included, since each training sample gets its augmentation seed from the batch sampler, whose state is in the checkpoint. `--epochs` may be raised when resuming, and the cosine schedule follows the new value. Resuming a run that has already finished only re-evaluates the selected checkpoint.

### Training further

The released `hannoree.pt` is a checkpoint like `best.pt`, so training can go on from it:

- **On new data**, start a new run from its weights and vocabulary: `hannoree train --init-from hannoree.pt --train ... --dev ...`. The run starts at epoch 1 with a fresh optimizer and the usual schedule, so lower `--lr`, `--finetune-lr` and `--epochs`, and consider `--finetune-from 1`. Tokens of the new data that are not in the vocabulary map to `<unk>`, and a warning is printed.
- **To continue the original run**, e.g. for more epochs, resume it with its optimizer state and learning-rate schedule: `hannoree train --resume hannoree.pt --epochs 400 --output-dir more_epochs`, with the original datasets. Its validation SER stays the one to beat, so use the original `--dev` datasets.

---

## Evaluation

The metric is the symbol error rate (SER) of the Zeus package: the edit distance between the predicted and the gold LMX token sequences, divided by the number of gold tokens.

**During training**, the validation SER is evaluated every `--evaluation-each` epochs and selects `best.pt`. With `--evaluation-each 0`, the validation loss selects it instead. After training, the selected checkpoint is evaluated on the dev and test splits. Validation loss, token accuracy, validation SER and test SER (overall and per dataset) are printed and saved in `results.json`.

**A trained model**, e.g. the released one, is evaluated with:

```bash
hannoree evaluate --model hannoree.pt --datasets omniomr dolores --split test --output results/test.json
```

It prints the SER of the split and of each dataset, and with `--output` saves them as JSON.

---

## Repository layout

```text
src/hannoree/
├── model.py          # MuSViT backbone, adapter (MusvitEncoder), Zeus decoder, HanNoReEModel
├── data.py           # image preprocessing, token vocabulary, loading of pickled Zeus splits, dataset, batching
├── augmentation.py   # Zeus training image augmentation (--augment), ported from TensorFlow
├── train.py          # hannoree train
├── evaluate.py       # SER evaluation (also during training), hannoree evaluate
├── inference.py      # load_model, hannoree transcribe
└── cli.py            # the hannoree command
scripts/train.slurm   # training of the released model
```

---

## Releasing a new model

This section is for maintainers. Every GitHub release carries the model as `hannoree.pt`, so the latest release always has the latest model. Its tag, e.g. `v1.2.0`, is also the package version. Bump the minor version for a new model, and the patch version for code changes only.

1. **Train** the model, e.g. with `scripts/train.slurm`, and check `experiment_results/results.json`.
2. **Verify** the checkpoint, e.g. that it reproduces the test SER of `results.json`:

   ```bash
   hannoree evaluate --model experiment_results/best.pt --datasets dolores olimpicscan olimpicsynth omniomr
   ```

3. **Rename** `experiment_results/best.pt` to `hannoree.pt`. A release asset may be up to 2 GiB, and the checkpoint is about 1.1 GB.
4. **Create the release** on GitHub: open the repository, *Releases* → *Draft a new release*.
   - *Choose a tag*: type the new tag, e.g. `v1.2.0`, and *Create new tag on publish*. Its target is `master`, which should hold the code the model was trained and verified with.
   - *Release title*: e.g. `HanNoReE 1.2.0`.
   - *Description*: what changed, and for a new model the training data (the `--train` and `--dev` datasets), any deviation from the default recipe, the epoch, and the validation and test SER per dataset from `results.json`. Mention that the model is licensed under CC BY-NC-SA 4.0, as a derivative of MuSViT whose terms also apply.
   - Drag `hannoree.pt` into *Attach binaries*, and wait until the upload finishes.
   - *Publish release*.
5. **Check** the published model as a user gets it:

   ```bash
   gh release download v1.2.0 --repo OmniOMR/HanNoReE --pattern hannoree.pt --dir /tmp/check
   hannoree transcribe --model /tmp/check/hannoree.pt some_stave.png
   ```

For a release with code changes only, attach the `hannoree.pt` of the previous release again, so the latest release keeps carrying the model.

---

## Citation

HanNoReE is built on the MuSViT backbone. Please also cite MuSViT when using HanNoReE or its released models:

```bibtex
@InProceedings{Penarrubia:ECCV:2026,
  author    = {Penarrubia, Carlos and Rios-Vila, Antonio and Fuentes-Martinez, Eliseo and Martinez-Sevilla, Juan C. and Castellanos, Francisco J. and Alfaro-Contreras, Mar{\'i}a and Calvo-Zaragoza, Jorge},
  editor    = {Favaro, Paolo and Kukelova, Zuzana and Maki, Atsuto and Rohrbach, Anna and Schindler, Konrad and Tombari, Federico},
  title     = {{MuSViT}: A Foundation Vision Model for Sheet Music Representation},
  booktitle = {Computer Vision -- ECCV 2026},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {459--476},
  isbn      = {978-3-032-37271-0},
  doi       = {10.1007/978-3-032-37271-0_25}
}
```

---

## License

The code and the released models are licensed under [CC BY-NC-SA 4.0](LICENSE), compatible with the license of [MuSViT](https://huggingface.co/PRAIG/musvit), from which the models are fine-tuned.
