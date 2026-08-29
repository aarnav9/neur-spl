#!/usr/bin/env python3
"""Validate research-critical fields in the development pilot config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


REQUIRED_SECTIONS = {
    "data",
    "model",
    "training",
    "diagnostic",
    "capacity",
    "dynamic_sparse_training",
    "gates",
}


def validate(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    if config["data"]["fit_examples"] + config["data"]["selection_examples"] != 50000:
        raise ValueError("CIFAR development split must contain exactly 50,000 examples")
    if config["data"]["official_test_role"] != "locked_final_only":
        raise ValueError("official test data must remain locked during development")
    if config["data"]["source"] not in {"torchvision", "huggingface"}:
        raise ValueError("data source must be torchvision or huggingface")
    if config["data"]["source"] == "huggingface" and not config["data"].get(
        "huggingface_revision"
    ):
        raise ValueError("the Hugging Face dataset revision must be pinned")
    if config["capacity"].get("flow_scope") != "two_conv_residual_branch":
        raise ValueError("the pilot flow scope must remain explicit")
    if int(config["capacity"].get("flow_channel_group_size", 0)) < 1:
        raise ValueError("flow_channel_group_size must be positive")
    seeds = config["training"]["seeds"]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("at least three unique seeds are required")
    if "cut_to_grow_shuffled_within_layer" not in config["dynamic_sparse_training"]["methods"]:
        raise ValueError("the within-layer shuffle sanity control is mandatory")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = validate(args.config)
    print(f"valid: {args.config} ({config['protocol_version']})")


if __name__ == "__main__":
    main()
