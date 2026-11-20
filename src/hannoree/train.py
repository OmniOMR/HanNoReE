"""
Training of the HanNoReE model on the pickled splits of Zeus datasets.

- Each split (train / dev / test) can combine any datasets found in the dataset directory.
- The MuSViT backbone is frozen for the first epochs, while the randomly initialized adapter and decoder learn to
  read its features, and fine-tuned from --finetune-from on.
- The validation SER (the metric of the Zeus package) selects the checkpoint; the selected checkpoint is then
  evaluated on the dev and test splits.

The defaults are the recipe of the released model (see scripts/train.slurm).
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .augmentation import ZeusAugmentation
from .data import StaveOMRDataset, TokenVocabulary, discover_datasets, load_split
from .evaluate import (
    evaluate_split,
    evaluate_teacher_forced,
    export_results,
    make_loader,
    print_ser,
    to_device,
)
from .inference import load_checkpoint
from .model import HanNoReEModel

MODEL_NAME = "HanNoReE (MuSViT + Zeus decoder)"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def learning_rate_at(epoch: int, args: argparse.Namespace) -> float:
    """
    Learning rate of a (1-based) epoch. With 'cos' decay, it follows a cosine from the
    initial rate down towards 1% of it over all epochs. Being a pure function of the epoch,
    it stays correct when training resumes, even with a changed --epochs.
    """
    if args.lr_decay != "cos":
        return args.learning_rate
    min_lr = args.learning_rate * 0.01
    progress = (epoch - 1) / max(1, args.epochs)
    return min_lr + (args.learning_rate - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def finetunes_backbone(epoch: int, args: argparse.Namespace) -> bool:
    """Whether the MuSViT backbone is trained in a (1-based) epoch."""
    return args.finetune_from is not None and epoch >= args.finetune_from


def backbone_learning_rate_at(epoch: int, args: argparse.Namespace, progress: float = 1.0) -> float:
    """
    Learning rate of the MuSViT backbone after `progress` (0-1] of a (1-based) epoch: 0 while frozen, then
    --finetune-lr scaled by the same schedule as the rest of the model, so that it decays along with it.
    During the first --finetune-warmup-epochs of fine-tuning, it rises linearly from 0, step by step, so that
    the first updates of the pre-trained weights, made before Adam's moment estimates settle, stay small.
    """
    if not finetunes_backbone(epoch, args):
        return 0.0
    lr = args.finetune_lr * learning_rate_at(epoch, args) / args.learning_rate
    if args.finetune_warmup_epochs > 0:
        lr *= min(1.0, (epoch - args.finetune_from + progress) / args.finetune_warmup_epochs)
    return lr


def stave_width_arg(value: str) -> int | None:
    """--stave-width: a width in pixels, or 'aspect' (None) to keep each stave's aspect ratio."""
    if value.lower() == "aspect":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a width in pixels or 'aspect', got '{value}'") from None


def finetune_from_arg(value: str) -> int | None:
    """--finetune-from: an epoch, or 'none' (None) to keep the backbone frozen."""
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an epoch or 'none', got '{value}'") from None


def model_config(vocab: TokenVocabulary, args: argparse.Namespace) -> dict:
    """
    What rebuilds the model for inference without the training data or arguments, saved in every checkpoint
    (see inference.py): the HanNoReEModel arguments under 'model' (the ViT configuration is added once the
    backbone is loaded) and the preprocessing of the input images under 'input'.
    """
    return {
        "model": {
            "vocab_size": len(vocab),
            "dim": args.dim,
            "bos_idx": vocab.bos_idx,
            "eos_idx": vocab.eos_idx,
            "pad_idx": vocab.pad_idx,
            "max_length": args.max_gen_length,
            "dropout": args.dropout,
            "musvit_model": args.musvit_model,
            "stave_height": args.stave_height,
            # The widest stave band, over which the position embeddings are stretched with 'interpolate'
            "max_stave_width": args.stave_width or args.max_stave_width,
        },
        "input": {
            "stave_height": args.stave_height,
            "stave_width": args.stave_width,
            "max_stave_width": args.max_stave_width,
        },
    }


def load_initial_weights(model: HanNoReEModel, state_dict: dict):
    """
    Loads the weights of another training run or a released model (--init-from). Weights without a MuSViT backbone
    leave it pre-trained; all other weights must match.
    """
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("encoder.backbone.")]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"The --init-from weights do not fit the model. "
            f"Missing weights: {missing}. Unexpected weights: {result.unexpected_keys}."
        )
    source = "pre-trained (not in --init-from)" if result.missing_keys else "from --init-from"
    print(f"Initialized the model from --init-from; MuSViT backbone weights: {source}.")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    vocab: TokenVocabulary,
    epoch: int,
    best: dict | None,
    history: list[dict],
    train_time_sec: float,
    config: dict,
):
    """Saves everything needed to evaluate the model or to resume its training exactly."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vocab": vocab.to_dict(),
            "config": config,
            "epoch": epoch,
            "best": best,
            "history": history,
            "train_time_sec": train_time_sec,
            "rng": get_rng_state(),
        },
        path,
    )


def resolve_checkpoint(args: argparse.Namespace) -> Path | None:
    """The checkpoint to resume from: an explicit --resume path, or with 'auto' the latest one of --output-dir."""
    if not args.resume:
        return None
    if args.resume.lower() != "auto":
        path = Path(args.resume)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint to resume from not found: '{path}'")
        return path
    output_dir = Path(args.output_dir)
    for candidate in (output_dir / "latest.pt", output_dir / "best.pt"):
        if candidate.is_file():
            return candidate
    print(f"No checkpoint found in '{output_dir}'; training from scratch.")
    return None


def selection_metric(args: argparse.Namespace) -> str:
    """The checkpoint is selected by validation SER; by validation loss only when SER is never evaluated."""
    return "ser" if args.evaluation_each > 0 else "val_loss"


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def describe_settings(args: argparse.Namespace) -> str:
    """The backbone settings, for the results."""
    tuning = f"fine-tuned from epoch {args.finetune_from}" if args.finetune_from is not None else "frozen"
    width = args.stave_width or f"aspect <= {args.max_stave_width}"
    return f"{args.musvit_model} ({args.stave_height} x {width}, {tuning})"


def build_model(
    vocab: TokenVocabulary,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint: dict | None = None,
) -> tuple[HanNoReEModel, dict]:
    """The model, and its configuration to save in checkpoints (see model_config)."""
    config = model_config(vocab, args)
    # A resumed checkpoint holds all backbone weights (frozen ones too), so the pre-trained ones are not downloaded
    resumed_vit_config = ((checkpoint or {}).get("config") or {}).get("model", {}).get("vit_config")
    model = HanNoReEModel(
        **config["model"],
        vit_config=resumed_vit_config,
        hf_token=args.hf_token,
        musvit_bf16=args.musvit_bf16,
        gradient_checkpointing=args.grad_checkpointing,
    ).to(device)

    # Builds the ViT without downloading the pre-trained weights, which the checkpoint replaces anyway
    backbone = model.backbone
    config["model"]["vit_config"] = backbone.vit.config.to_dict()

    enc_params, dec_params = n_params(model.encoder), n_params(model.decoder)
    print(
        f"Parameters: Encoder: {enc_params:,} (MuSViT backbone: {n_params(backbone):,}) | "
        f"Decoder: {dec_params:,} | Total: {enc_params + dec_params:,}"
    )
    return model, config


def train(
    model: HanNoReEModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab: TokenVocabulary,
    args: argparse.Namespace,
    device: torch.device,
    config: dict,
    checkpoint: dict | None = None,
    init_weights: dict | None = None,
) -> tuple[dict, list[dict], float]:
    """
    Trains the model and loads the weights of the selected epoch. `checkpoint` resumes a run exactly;
    `init_weights` only provides the initial weights of a new run. Returns the metrics of the selected epoch
    (without those not evaluated during training), the history of all epochs, and the total training time.
    """
    backbone = model.backbone

    output_dir = Path(args.output_dir)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    epochs_dir = output_dir / "checkpoints"
    if args.save_epoch_checkpoints:
        epochs_dir.mkdir(parents=True, exist_ok=True)

    # Label smoothing applies to the training loss only; the validation loss stays plain cross-entropy,
    # comparable across runs
    train_criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=args.label_smoothing)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # Optimizer as in TensorFlow Zeus; the fused CUDA kernel updates all parameters at once.
    # The MuSViT backbone gets a second parameter group with its own learning rate, from the start, so that the
    # optimizer state keeps one layout whether the backbone is frozen or not (frozen weights get no gradients,
    # which the optimizer skips).
    backbone_ids = {id(p) for p in backbone.parameters()}
    param_groups = [
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids]},
        {"params": list(backbone.parameters())},
    ]
    fused = device.type == "cuda"
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=args.learning_rate, eps=1e-7, fused=fused)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay, fused=fused)

    select_by = selection_metric(args)

    start_epoch = 1
    best: dict | None = None
    history: list[dict] = []
    prior_train_time = 0.0

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best = checkpoint["best"]
        history = checkpoint["history"]
        prior_train_time = checkpoint["train_time_sec"]
        set_rng_state(checkpoint["rng"])
        print(f"Resumed from epoch {start_epoch - 1}.")
        # Resuming a checkpoint from elsewhere (e.g. a released model) into a new --output-dir: the selected weights
        # must be there too. A best checkpoint is them; otherwise they are unavailable, and selection starts anew.
        if best is not None and not best_path.is_file():
            if best["epoch"] == checkpoint["epoch"]:
                save_checkpoint(best_path, model, optimizer, vocab, checkpoint["epoch"], best, history, prior_train_time, config)
            else:
                best = None
        if history and history[-1].get("backbone_lr") and not finetunes_backbone(start_epoch, args):
            print(
                f"Warning: the backbone was fine-tuned before epoch {start_epoch}, "
                f"but stays frozen from now on, since --finetune-from is 'none' or later."
            )
    elif init_weights is not None:
        load_initial_weights(model, init_weights)

    if start_epoch > args.epochs:
        print(f"Model already reached epoch {start_epoch - 1} >= requested target epochs {args.epochs}. Skipping training loop.")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        lr = learning_rate_at(epoch, args)
        optimizer.param_groups[0]["lr"] = lr
        backbone_lr = None
        finetune = finetunes_backbone(epoch, args)
        if finetune != backbone.trainable:
            print(f"Epoch {epoch}: {'unfreezing' if finetune else 'freezing'} the MuSViT backbone.")
        backbone.set_trainable(finetune)

        model.train()
        # Accumulated on the device, so that the CPU does not wait for the GPU after every batch
        train_loss_sum = torch.zeros((), device=device)
        train_tokens = torch.zeros((), dtype=torch.long, device=device)

        steps = len(train_loader)
        for step, batch in enumerate(train_loader):
            inputs, lengths, input_seqs, targets = to_device(batch, device)
            # Set at every step, so that the warmup after unfreezing rises smoothly
            backbone_lr = backbone_learning_rate_at(epoch, args, (step + 1) / steps)
            optimizer.param_groups[1]["lr"] = backbone_lr

            optimizer.zero_grad()
            logits = model(inputs, input_seqs, lengths)  # (B, L, V)

            loss = train_criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n_tokens = (targets != vocab.pad_idx).sum()
            train_loss_sum += loss.detach() * n_tokens
            train_tokens += n_tokens

        train_loss = train_loss_sum.item() / max(1, int(train_tokens))
        val_loss, token_acc = evaluate_teacher_forced(model, val_loader, criterion, vocab, device)

        # Periodic Symbol Error Rate (SER) evaluation, as in Zeus
        should_eval_ser = (
            args.evaluation_each > 0
            and epoch >= args.evaluation_from
            and (epoch % args.evaluation_each == 0 or epoch == args.epochs)
        )
        current_ser = evaluate_split(model, val_loader, vocab, device)[0] if should_eval_ser else None

        record = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "token_acc": token_acc,
            "ser": current_ser,
            "sec": time.time() - epoch_start,
        }
        if backbone_lr is not None:
            record["backbone_lr"] = backbone_lr
        history.append(record)
        train_time = prior_train_time + time.time() - start_time

        improved = record[select_by] is not None and (best is None or record[select_by] < best[select_by])
        if improved:
            best = {k: record[k] for k in ("epoch", "train_loss", "val_loss", "token_acc", "ser")}
            save_checkpoint(best_path, model, optimizer, vocab, epoch, best, history, train_time, config)

        save_checkpoint(latest_path, model, optimizer, vocab, epoch, best, history, train_time, config)

        if args.save_epoch_checkpoints and should_eval_ser:
            save_checkpoint(
                epochs_dir / f"epoch_{epoch:03d}.pt", model, optimizer, vocab, epoch, best, history, train_time, config
            )

        ser_str = f" | Val SER: {current_ser:.2f}%" if current_ser is not None else ""
        backbone_lr_str = f" | Backbone LR: {backbone_lr:.2e}" if backbone_lr else ""
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"LR: {lr:.2e}{backbone_lr_str} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Token Acc: {token_acc:.2f}%"
            f"{ser_str} | "
            f"Time: {record['sec']:.1f}s"
            f"{' | * best' if improved else ''}"
        )

    total_training_time = prior_train_time + time.time() - start_time

    # All validation and test metrics are reported from one set of weights: the selected checkpoint
    if best is not None and best_path.is_file():
        print(f"\nLoading the selected checkpoint (epoch {best['epoch']}, by {select_by}) from: {best_path}")
        model.load_state_dict(load_checkpoint(best_path)["model"])
    else:
        print("\nNo checkpoint was selected during training; evaluating the final weights.")
        if history:
            best = {"epoch": history[-1]["epoch"], "train_loss": history[-1]["train_loss"]}
        else:
            best = {"epoch": start_epoch - 1}
    return best, history, total_training_time


def evaluate_selected(
    model: HanNoReEModel,
    best: dict,
    val_loader: DataLoader,
    test_dataset: StaveOMRDataset | None,
    vocab: TokenVocabulary,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """
    Completes the validation metrics of the selected weights, whose known ones `best` holds, and computes the
    test SER, also per dataset. Returns the rounded metrics.
    """

    if best.get("val_loss") is None:
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        best["val_loss"], best["token_acc"] = evaluate_teacher_forced(model, val_loader, criterion, vocab, device)
    if best.get("ser") is None:
        best["ser"] = evaluate_split(model, val_loader, vocab, device)[0]
    print(
        f"Selected checkpoint: epoch {best['epoch']} | Val Loss: {best['val_loss']:.4f} | "
        f"Token Acc: {best['token_acc']:.2f}% | Val SER: {best['ser']:.2f}%"
    )

    test_ser = None
    test_ser_per_dataset: dict[str, float] = {}
    if test_dataset is not None and len(test_dataset) > 0:
        print("\n=======================================================")
        print(" Running Test Set Evaluation...")
        print("=======================================================")
        test_loader = make_loader(test_dataset, args.batch_size, args.num_workers, shuffle=False)
        test_ser, per_dataset = evaluate_split(model, test_loader, vocab, device)
        print_ser("Final Test", test_ser, per_dataset)
        if len(per_dataset) > 1:
            test_ser_per_dataset = per_dataset

    return {
        "best_epoch": best["epoch"],
        "train_loss": round(best["train_loss"], 4) if best.get("train_loss") is not None else None,
        "val_loss": round(best["val_loss"], 4),
        "token_acc": round(best["token_acc"], 2),
        "ser": round(best["ser"], 2),
        "test_ser": round(test_ser, 2) if test_ser is not None else None,
        "test_ser_per_dataset": {k: round(v, 2) for k, v in test_ser_per_dataset.items()},
    }


# ==============================================================================
# Command line
# ==============================================================================

def define_parser(parser: argparse.ArgumentParser):
    data = parser.add_argument_group("datasets")
    data.add_argument("--dataset-dir", type=str, default="datasets", help="Directory containing the datasets as subfolders (default: datasets).")
    data.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="Datasets used for all splits unless overridden by --train/--dev/--test (default: all datasets in --dataset-dir).",
    )
    dataset_spec_help = (
        "Dataset(s) whose predefined '{}' split to use: names of subfolders of --dataset-dir, "
        "dataset folders, or .pickle files (default: {})."
    )
    data.add_argument("--train", type=str, nargs="+", default=None, help=dataset_spec_help.format("train", "--datasets"))
    data.add_argument(
        "--dev", "--val", dest="dev", type=str, nargs="+", default=None,
        help=dataset_spec_help.format("dev", "--datasets, or the --train datasets if --datasets is not given"),
    )
    data.add_argument(
        "--test", type=str, nargs="*", default=None,
        help=dataset_spec_help.format("test", "--datasets, or the --train datasets if --datasets is not given")
             + " Pass --test without values to skip testing.",
    )

    musvit = parser.add_argument_group("MuSViT backbone and input")
    musvit.add_argument("--musvit-model", type=str, default="PRAIG/musvit", help="Hugging Face ID of the ViT backbone, e.g. 'PRAIG/musvit' or 'PRAIG/musvit-light' (loaded as a plain ViTModel; gated, see --hf-token).")
    musvit.add_argument("--stave-height", type=int, default=96, help="Height staves are resized to, a multiple of 16 (default: 96).")
    musvit.add_argument("--stave-width", type=stave_width_arg, default=1024, help="Width staves are resized to, a multiple of 16 (default: 1024). 'aspect' keeps each stave's aspect ratio: the width follows the stave height, rounded to a multiple of 16 and capped at --max-stave-width; batches are padded with white and the padding is masked.")
    musvit.add_argument("--max-stave-width", type=int, default=2048, help="Cap of the width of staves with --stave-width aspect; longer staves are squashed. The pre-trained position embeddings are then stretched over this width.")
    musvit.add_argument("--musvit-bf16", action="store_true", help="Run the backbone in bfloat16 autocast (Ampere+ GPUs): much faster and less memory, slightly less precise.")
    musvit.add_argument("--grad-checkpointing", action="store_true", help="Recompute the backbone's activations in the backward pass instead of storing them, to fine-tune with less GPU memory at ~30%% more compute.")
    musvit.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for the gated MuSViT models (default: the HF_TOKEN environment variable).")

    finetune = parser.add_argument_group("backbone fine-tuning")
    finetune.add_argument("--finetune-from", type=finetune_from_arg, default=20, metavar="EPOCH", help="Fine-tune all backbone weights from this (1-based) epoch on, frozen before; 1 fine-tunes from the start, 'none' keeps it frozen (default: 20). Applies to resumed runs as well.")
    finetune.add_argument("--finetune-lr", type=float, default=1e-5, help="Learning rate of the backbone while fine-tuning, at the schedule's peak; it follows the --lr-decay schedule of --learning-rate, scaled. Default: 1e-5.")
    finetune.add_argument("--finetune-warmup-epochs", type=float, default=5, help="The backbone learning rate rises linearly from 0 over this many epochs (fractions allowed) after unfreezing, updated at every step; 0 disables the warmup (default: 5).")

    training = parser.add_argument_group("training")
    training.add_argument("--output-dir", type=str, default="experiment_results", help="Directory for checkpoints and results (default: experiment_results).")
    training.add_argument("--init-from", type=str, default=None, help="Start a new training run (epoch 1, fresh optimizer) from the weights of a checkpoint, e.g. to fine-tune a released model on new data; its vocabulary is used. Ignored when a checkpoint is resumed.")
    training.add_argument("--resume", type=str, default=None, help="Resume training: 'auto' (the latest checkpoint of --output-dir, if any) or the path of a checkpoint.")
    training.add_argument("--epochs", type=int, default=300, help="Number of training epochs (default: 300).")
    training.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32).")
    training.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3, help="Learning rate of the adapter and decoder (default: 1e-3).")
    training.add_argument("--lr-decay", type=str, default="cos", choices=["cos", "none"], help="Learning rate decay (default: cos).")
    training.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="Optimizer (default: adam, as Zeus).")
    training.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (adamw only).")
    training.add_argument("--dim", type=int, default=256, help="Model hidden / embedding dimension (default: 256).")
    training.add_argument("--dropout", type=float, default=0.2, help="Dropout rate of the adapter (default: 0.2).")
    training.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing of the training loss; the validation loss is computed without it (default: 0.1).")
    training.add_argument("--augment", type=str, default="h:8,rotate:1,v:4", help="Training image augmentation, with the syntax of Zeus --augment, e.g. 'h:8,rotate:1,v:4,de,en3:0.2,n:0.01,c:-1:1,b:-0.5:0.2' (see augmentation.py); pixel amounts refer to the stave band. Default: 'h:8,rotate:1,v:4'; '' disables it.")
    training.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    training.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    training.add_argument("--max-gen-length", type=int, default=None, help="Max length of autoregressive decoding (default: 1.2x the longest training sequence).")
    training.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu'). Auto-detected if not specified.")
    training.add_argument("--no-tf32", action="store_true", help="Compute matrix multiplications in full FP32 instead of TF32 (TF32 runs on the tensor cores of Ampere and newer GPUs).")

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--evaluation-from", type=int, default=1, help="Evaluate the validation SER from this epoch on.")
    evaluation.add_argument("--evaluation-each", "--eval-interval", dest="evaluation_each", type=int, default=1, help="Evaluate the validation SER every N epochs (default: 1). It selects the best checkpoint; 0 selects by validation loss instead.")
    evaluation.add_argument("--save-epoch-checkpoints", action="store_true", help="Also keep a checkpoint of every epoch the validation SER is evaluated, in <output-dir>/checkpoints.")

    # Old settings that some previously trained models used, which HanNoReE does not support
    # Accepted and ignored.
    for flag in ("--model", "--musvit-input", "--feature-layout", "--finetune-method", "--lora-rank", "--lora-alpha", "--lora-dropout"):
        parser.add_argument(flag, help=argparse.SUPPRESS)


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace):
    if args.finetune_from is not None and args.finetune_from < 1:
        parser.error("--finetune-from must be an epoch >= 1, or 'none'.")
    if args.finetune_warmup_epochs < 0:
        parser.error("--finetune-warmup-epochs must be >= 0.")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0, 1).")
    for name in ("stave_width", "max_stave_width"):
        width = getattr(args, name)
        if width is not None and (width <= 0 or width % 16 != 0):
            parser.error(f"--{name.replace('_', '-')} must be a positive multiple of 16 (the patch size), got {width}.")
    try:
        ZeusAugmentation(args.augment)
    except ValueError as e:
        parser.error(f"--augment: {e}")


def run(args: argparse.Namespace):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Active Device: {device}")

    # TF32 tensor cores for matrix multiplications (decoder LSTM cell, attention, projections); PyTorch already
    # uses TF32 for convolutions and cuDNN LSTMs by default. No effect on the CPU or pre-Ampere GPUs.
    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Select datasets for each split and load their predefined splits.
    # Without --datasets, dev and test follow the training datasets rather than everything in --dataset-dir.
    train_specs = args.train or args.datasets or discover_datasets(args.dataset_dir)
    default_eval_specs = args.datasets or train_specs
    dev_specs = args.dev or default_eval_specs
    test_specs = args.test if args.test is not None else default_eval_specs

    if not train_specs:
        raise FileNotFoundError(
            f"No datasets selected and none found in '{args.dataset_dir}'.\n"
            f"To convert a MusiCorpus dataset to a pickled Zeus dataset, run:\n"
            f"  zeus musicorpus --input <mc_dataset> --output {args.dataset_dir}/<name> --take-staves\n"
            f"  zeus pickle {args.dataset_dir}/<name>/samples.<split>.txt   # for each split"
        )

    print("Loading training datasets...")
    train_samples = load_split(train_specs, args.dataset_dir, "train")
    print("Loading validation datasets...")
    val_samples = load_split(dev_specs, args.dataset_dir, "dev")
    print("Loading test datasets...")
    test_samples = load_split(test_specs, args.dataset_dir, "test") if test_specs else {}

    selection = {"train": list(train_samples), "dev": list(val_samples), "test": list(test_samples)}
    for split, names in selection.items():
        print(f"  {split}: {', '.join(names) or '-'}")

    all_train = [s for samples in train_samples.values() for s in samples]
    n_val = sum(map(len, val_samples.values()))
    n_test = sum(map(len, test_samples.values()))
    if n_val == 0:
        raise ValueError("The validation split is empty; it is needed for checkpoint selection.")
    print(f"Data Split: {len(all_train):,} Train | {n_val:,} Validation | {n_test:,} Test")

    # Decoding must be able to produce the longest transcriptions, or SER is inflated by truncation
    max_train_len = max(len(s.lmx.split()) for s in all_train)
    if args.max_gen_length is None:
        args.max_gen_length = int(1.2 * max_train_len) + 1
    print(f"Max generation length: {args.max_gen_length} (longest training sequence: {max_train_len} tokens)")
    for split, samples in (("dev", val_samples), ("test", test_samples)):
        too_long = sum(len(s.lmx.split()) > args.max_gen_length for ss in samples.values() for s in ss)
        if too_long:
            print(f"Warning: {too_long:,} {split} sample(s) are longer than the max generation length and will be truncated.")

    checkpoint_path = resolve_checkpoint(args)
    checkpoint = None
    if checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
    init_weights, init_vocab = None, None
    if args.init_from:
        if checkpoint is not None:
            print(f"Resuming '{checkpoint_path}', so --init-from '{args.init_from}' is ignored.")
        else:
            print(f"Initializing weights from: {args.init_from}")
            init_checkpoint = load_checkpoint(args.init_from)
            init_weights, init_vocab = init_checkpoint["model"], init_checkpoint["vocab"]

    # 2. Vocabulary from the training split (as Zeus TokenMap), or the exact mapping of the checkpoint / --init-from
    vocab_dict = checkpoint["vocab"] if checkpoint is not None else init_vocab
    if vocab_dict is not None:
        vocab = TokenVocabulary.from_dict(vocab_dict)
        unknown = {t for s in all_train for t in s.lmx.split()} - vocab.token2id.keys()
        if unknown:
            print(f"Warning: {len(unknown):,} training token type(s) are not in the model's vocabulary and map to <unk>.")
    else:
        vocab = TokenVocabulary.build_from_samples(all_train)
    vocab.save(output_dir / "vocab.json")
    print(f"Vocabulary: {len(vocab):,} unique LMX tokens.")

    dataset_kwargs = dict(
        vocab=vocab,
        stave_height=args.stave_height,
        stave_width=args.stave_width,
        max_stave_width=args.max_stave_width,
    )
    train_ds = StaveOMRDataset(train_samples, **dataset_kwargs, augment=args.augment)
    val_ds = StaveOMRDataset(val_samples, **dataset_kwargs)
    test_ds = StaveOMRDataset(test_samples, **dataset_kwargs) if test_samples else None

    # 3. Train and evaluate the selected weights
    print("\n=======================================================")
    print(f" Training {MODEL_NAME}")
    print("=======================================================")

    set_seed(args.seed)
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False, seed=args.seed)
    model, config = build_model(vocab, args, device, checkpoint)

    best, history, train_time = train(model, train_loader, val_loader, vocab, args, device, config, checkpoint, init_weights)
    metrics = evaluate_selected(model, best, val_loader, test_ds, vocab, args, device)

    enc_params, dec_params = n_params(model.encoder), n_params(model.decoder)
    export_results(
        {
            "model": MODEL_NAME,
            "settings": describe_settings(args),
            "enc_params": enc_params,
            "dec_params": dec_params,
            "total_params": enc_params + dec_params,
            "select_by": selection_metric(args),
            **metrics,
            "epochs_trained": len(history),
            "total_time_sec": round(train_time, 1),
            "avg_epoch_sec": round(train_time / max(1, len(history)), 2),
            "datasets": selection,
            "augment": args.augment,
        },
        output_dir,
    )
