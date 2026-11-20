"""The `hannoree` command line tool (also `python -m hannoree`)."""

from __future__ import annotations

import argparse

from . import evaluate, inference, train

# name: (description, define_parser, run)
COMMANDS = {
    "train": ("Train the model on pickled Zeus datasets.", train.define_parser, train.run),
    "evaluate": ("Evaluate a trained model (SER) on the split of some datasets.", evaluate.define_parser, evaluate.run),
    "transcribe": ("Transcribe stave images into LMX and MusicXML.", inference.define_parser, inference.run),
}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="hannoree", description="HanNoReE: staff-level OMR with MuSViT and a Zeus decoder.")
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)
    for name, (description, define_parser, _) in COMMANDS.items():
        define_parser(subparsers.add_parser(name, description=description, help=description))

    args = parser.parse_args(argv)
    if args.command == "train":
        train.validate_args(parser, args)
    COMMANDS[args.command][2](args)


if __name__ == "__main__":
    main()
