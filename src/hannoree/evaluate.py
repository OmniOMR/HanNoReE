"""
Evaluation: the Symbol Error Rate (SER, the metric of the Zeus package) of greedy decoding against the gold LMX,
over a split and per source dataset.

Used during and at the end of training, and on its own for a trained model:

    hannoree evaluate --model hannoree.pt --datasets omniomr dolores
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from zeus.evaluation.symbol_error_rate import symbol_error_rate, tokenize

from .data import StaveBatchSampler, StaveCollate, StaveOMRDataset, TokenVocabulary, load_split


def make_loader(dataset: StaveOMRDataset, batch_size: int, num_workers: int, shuffle: bool, seed: int = 42) -> DataLoader:
    """
    Random batches for training (shuffle=True), augmented if the dataset has an augmentation; batches in dataset
    order for evaluation.
    """
    vocab = dataset.vocab
    return DataLoader(
        dataset,
        batch_sampler=StaveBatchSampler(
            len(dataset), batch_size, shuffle=shuffle, sample_seeds=shuffle and dataset.augmentation is not None
        ),
        collate_fn=StaveCollate(pad_idx=vocab.pad_idx, bos_idx=vocab.bos_idx, eos_idx=vocab.eos_idx),
        num_workers=num_workers,
        # Workers live across epochs instead of being re-forked (with the whole dataset) for every pass
        persistent_workers=num_workers > 0,
        # Page-locked batches can be copied to the GPU asynchronously
        pin_memory=torch.cuda.is_available(),
        # The worker seed comes from this generator instead of the global RNG. Persistent workers draw it only
        # when first started, which would otherwise shift the shuffling of a resumed run against an uninterrupted one.
        generator=torch.Generator().manual_seed(seed),
    )


def to_device(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Moves a collated batch to the device. Input lengths stay on the CPU, where sequence packing needs them."""
    inputs, lengths, input_seqs, targets = batch
    return (
        inputs.to(device, non_blocking=True),
        lengths,
        input_seqs.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
    )


def evaluate_teacher_forced(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    vocab: TokenVocabulary,
    device: torch.device,
) -> tuple[float, float]:
    """Per-token loss and token accuracy (%) under teacher forcing."""
    model.eval()
    # Accumulated on the device, so that the CPU does not wait for the GPU after every batch
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = torch.zeros((), dtype=torch.long, device=device)
    with torch.no_grad():
        for batch in loader:
            inputs, lengths, input_seqs, targets = to_device(batch, device)
            logits = model(inputs, input_seqs, lengths)
            mask = targets != vocab.pad_idx
            n_tokens = mask.sum()
            loss_sum += criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)) * n_tokens
            correct += ((logits.argmax(dim=-1) == targets) & mask).sum()
            total += n_tokens
    total_tokens = max(1, int(total))
    return loss_sum.item() / total_tokens, 100.0 * int(correct) / total_tokens


def predict_lmx(model: nn.Module, loader: DataLoader, vocab: TokenVocabulary, device: torch.device) -> list[str]:
    """Greedy autoregressive predictions as LMX strings, in the order of the loader's dataset (an unshuffled loader)."""
    assert not loader.batch_sampler.shuffle, "Predictions need the dataset order of an unshuffled loader."
    model.eval()
    predictions: list[str] = []
    for batch in loader:
        inputs, lengths, _, _ = to_device(batch, device)
        predictions += [vocab.decode_prediction(ids) for ids in model.generate(inputs, lengths).cpu().tolist()]
    return predictions


def split_ser(dataset: StaveOMRDataset, preds: list[str]) -> tuple[float, dict[str, float]]:
    """
    SER of a split and of each of its source datasets. Each sample is compared only once: the SER of the split is
    the per-dataset ones weighted by their gold token counts, which is what the corpus-level SER sums to.
    """
    names, gold = dataset.dataset_names(), dataset.gold_lmx()
    per_dataset: dict[str, float] = {}
    gold_tokens: dict[str, int] = {}
    for name in dict.fromkeys(names):
        idx = [i for i, n in enumerate(names) if n == name]
        per_dataset[name] = symbol_error_rate([gold[i] for i in idx], [preds[i] for i in idx])
        gold_tokens[name] = sum(len(tokenize(gold[i])) for i in idx)
    ser = sum(per_dataset[name] * gold_tokens[name] for name in per_dataset) / sum(gold_tokens.values())
    return ser, per_dataset


def evaluate_split(
    model: nn.Module, loader: DataLoader, vocab: TokenVocabulary, device: torch.device
) -> tuple[float, dict[str, float]]:
    """SER of the split of an unshuffled loader, and per source dataset."""
    return split_ser(loader.dataset, predict_lmx(model, loader, vocab, device))


def print_ser(split: str, ser: float, per_dataset: dict[str, float]):
    print(f"{split} SER: {ser:.2f}%")
    if len(per_dataset) > 1:
        for name, value in per_dataset.items():
            print(f"  - {name}: {value:.2f}%")


def export_results(result: dict, output_dir: Path):
    """Writes the results of a training run as JSON and as a one-row table to Markdown (also printed) and CSV."""
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    def fmt_pct(value) -> str:
        return f"{value:.2f}%" if value is not None else "N/A"

    per_dataset = result["test_ser_per_dataset"]
    headers = [
        "Model", "Enc Params", "Dec Params", "Total Params", "Best Epoch",
        "Val Loss", "Token Acc (%)", "Val SER (%)", "Test SER (%)",
        *(f"Test SER {name} (%)" for name in per_dataset),
        "Train Time (s)",
    ]
    row = [
        result["model"],
        f"{result['enc_params']:,}",
        f"{result['dec_params']:,}",
        f"{result['total_params']:,}",
        str(result["best_epoch"]),
        f"{result['val_loss']:.4f}",
        fmt_pct(result["token_acc"]),
        fmt_pct(result["ser"]),
        fmt_pct(result["test_ser"]),
        *(fmt_pct(v) for v in per_dataset.values()),
        f"{result['total_time_sec']:.1f}s",
    ]

    markdown = "".join("| " + " | ".join(cells) + " |\n" for cells in (headers, ["---"] * len(headers), row))
    print("\n" + markdown)
    md_path = output_dir / "results_table.md"
    md_path.write_text(markdown, encoding="utf-8")

    csv_path = output_dir / "results_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(row)

    print("Results saved:")
    print(f"  - Markdown: {md_path}")
    print(f"  - CSV:      {csv_path}")


# ==============================================================================
# Command line: evaluate a trained model
# ==============================================================================

def define_parser(parser):
    parser.add_argument("--model", type=Path, required=True, help="The trained model (a training checkpoint, .pt).")
    parser.add_argument("--dataset-dir", default="datasets", help="Directory with the datasets as subfolders (default: datasets).")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Datasets to evaluate on: names of subfolders of --dataset-dir, dataset folders, or .pickle files.")
    parser.add_argument("--split", default="test", choices=["dev", "test"], help="Split of the datasets to evaluate on (default: test).")
    parser.add_argument("--output", type=Path, default=None, help="Write the SER as JSON to this file.")
    parser.add_argument("--batch-size", type=int, default=32, help="Staves per forward pass.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--device", help="'cuda' or 'cpu' (default: cuda if available).")
    parser.add_argument("--bf16", action="store_true", help="Run the MuSViT backbone in bfloat16 autocast (Ampere+ GPUs).")


def run(args):
    from .inference import load_model

    transcriber = load_model(args.model, args.device, args.bf16)
    print(f"Evaluating '{args.model}' (epoch {transcriber.epoch}) on {transcriber.device}.")
    samples = load_split(args.datasets, args.dataset_dir, args.split)
    dataset = StaveOMRDataset(samples, transcriber.vocab, **transcriber.input)

    max_length = transcriber.model.decoder.max_length
    too_long = sum(n > max_length for n in dataset.target_lengths())
    if too_long:
        print(f"Warning: {too_long:,} sample(s) are longer than the max generation length ({max_length}) and will be truncated.")

    loader = make_loader(dataset, args.batch_size, args.num_workers, shuffle=False)
    ser, per_dataset = evaluate_split(transcriber.model, loader, transcriber.vocab, transcriber.device)
    print_ser(args.split, ser, per_dataset)

    if args.output is not None:
        result = {
            "model": str(args.model),
            "epoch": transcriber.epoch,
            "split": args.split,
            "samples": len(dataset),
            "ser": round(ser, 2),
            "ser_per_dataset": {k: round(v, 2) for k, v in per_dataset.items()},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Results saved to '{args.output}'.")
