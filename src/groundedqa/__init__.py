"""Grounded question answering: LoRA fine-tuning, hybrid retrieval, calibrated abstention."""

__version__ = "0.2.0"


def main() -> None:
    from groundedqa.cli import main as cli_main

    cli_main()
