"""Deterministic data and training utilities for the CIFAR-100 pilot."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from .model import CifarResNet, cifar_resnet18


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
CIFAR100_HF_REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stratified_split_indices(
    labels: Sequence[int], selection_examples: int, seed: int
) -> tuple[list[int], list[int]]:
    """Return fit/selection indices with deterministic proportional stratification."""

    if not 0 < selection_examples < len(labels):
        raise ValueError("selection_examples must be between zero and dataset size")
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[int(label)].append(index)
    if len(by_class) < 2:
        raise ValueError("stratified splitting requires at least two classes")

    rng = random.Random(seed)
    total = len(labels)
    quotas: dict[int, int] = {}
    fractional: list[tuple[float, int]] = []
    for label, indices in by_class.items():
        rng.shuffle(indices)
        exact = selection_examples * len(indices) / total
        base = min(len(indices) - 1, math.floor(exact))
        quotas[label] = base
        fractional.append((exact - base, label))

    remainder = selection_examples - sum(quotas.values())
    for _, label in sorted(fractional, reverse=True):
        if remainder == 0:
            break
        if quotas[label] < len(by_class[label]) - 1:
            quotas[label] += 1
            remainder -= 1
    if remainder:
        raise ValueError("could not allocate the requested stratified selection split")

    fit: list[int] = []
    selection: list[int] = []
    for label in sorted(by_class):
        cutoff = quotas[label]
        selection.extend(by_class[label][:cutoff])
        fit.extend(by_class[label][cutoff:])
    rng.shuffle(fit)
    rng.shuffle(selection)
    return fit, selection


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def build_cifar100_loaders(
    root: str | Path,
    *,
    split_seed: int,
    selection_examples: int,
    batch_size: int,
    num_workers: int = 2,
    download: bool = False,
    source: str = "torchvision",
) -> tuple[DataLoader, DataLoader]:
    """Build augmented fit and unaugmented selection loaders.

    The official test set is intentionally not constructed here so it cannot be
    used accidentally during method development.
    """

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("torchvision is required for CIFAR-100") from exc

    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    fit_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    selection_transform = transforms.Compose([transforms.ToTensor(), normalize])
    root_path = Path(root)
    if source == "torchvision":
        augmented: Dataset = datasets.CIFAR100(
            root=str(root_path), train=True, download=download, transform=fit_transform
        )
        clean: Dataset = datasets.CIFAR100(
            root=str(root_path),
            train=True,
            download=False,
            transform=selection_transform,
        )
        labels = augmented.targets  # type: ignore[attr-defined]
    elif source == "huggingface":
        try:
            from datasets import load_dataset, load_from_disk
        except ImportError as exc:  # pragma: no cover - optional backend
            raise RuntimeError("install the project with code[hf] for Hugging Face data") from exc
        saved_path = root_path / "cifar100_hf_train"
        if saved_path.exists():
            dataset = load_from_disk(str(saved_path))
        elif download:
            dataset = load_dataset(
                "uoft-cs/cifar100",
                split="train",
                revision=CIFAR100_HF_REVISION,
                cache_dir=str(root_path / "huggingface"),
            )
            dataset.save_to_disk(str(saved_path))
        else:
            raise FileNotFoundError(
                f"missing {saved_path}; run make pull-cifar100-hf PYTHON=.venv/bin/python"
            )
        labels = [int(label) for label in dataset["fine_label"]]
        augmented = _HuggingFaceCifar100(dataset, fit_transform)
        clean = _HuggingFaceCifar100(dataset, selection_transform)
    else:
        raise ValueError("source must be 'torchvision' or 'huggingface'")

    fit_indices, selection_indices = stratified_split_indices(
        labels, selection_examples, split_seed
    )
    generator = torch.Generator().manual_seed(split_seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
    }
    fit_loader = DataLoader(
        Subset(augmented, fit_indices), shuffle=True, generator=generator, **common
    )
    selection_loader = DataLoader(Subset(clean, selection_indices), shuffle=False, **common)
    return fit_loader, selection_loader


def load_cifar100_coarse_mapping(root: str | Path) -> tuple[int, ...]:
    """Read the pinned fine-to-coarse class mapping from the local HF dataset."""

    try:
        from datasets import load_from_disk
    except ImportError as exc:  # pragma: no cover - optional backend
        raise RuntimeError("Hugging Face datasets is required for coarse labels") from exc
    dataset_path = Path(root) / "cifar100_hf_train"
    if not dataset_path.exists():
        raise FileNotFoundError(f"missing pinned CIFAR-100 dataset: {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    mapping: list[int | None] = [None] * 100
    for fine, coarse in zip(
        dataset["fine_label"], dataset["coarse_label"], strict=True
    ):
        fine_id = int(fine)
        coarse_id = int(coarse)
        previous = mapping[fine_id]
        if previous is not None and previous != coarse_id:
            raise ValueError(f"fine class {fine_id} maps to multiple coarse classes")
        mapping[fine_id] = coarse_id
    if any(value is None for value in mapping):
        raise ValueError("coarse mapping does not cover all 100 fine classes")
    output = tuple(int(value) for value in mapping if value is not None)
    if set(output) != set(range(20)):
        raise ValueError("coarse mapping does not contain the declared 20 commodities")
    return output


class _HuggingFaceCifar100(Dataset):
    def __init__(self, dataset: Any, transform: Any) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.dataset[index]
        return self.transform(record["img"]), int(record["fine_label"])


def build_synthetic_loaders(
    *, examples: int = 64, num_classes: int = 10, batch_size: int = 16, seed: int = 7
) -> tuple[DataLoader, DataLoader]:
    """Small deterministic loaders for wiring tests; never a research benchmark."""

    if examples < 2 * num_classes:
        raise ValueError("synthetic smoke data need at least two examples per class")
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(examples, 3, 32, 32, generator=generator)
    labels = torch.arange(examples) % num_classes
    fit_indices, selection_indices = stratified_split_indices(
        labels.tolist(), max(num_classes, examples // 5), seed
    )
    dataset = TensorDataset(images, labels)
    fit_loader = DataLoader(
        Subset(dataset, fit_indices), batch_size=batch_size, shuffle=True, generator=generator
    )
    selection_loader = DataLoader(
        Subset(dataset, selection_indices), batch_size=batch_size, shuffle=False
    )
    return fit_loader, selection_loader


def _batch_metrics(logits: Tensor, targets: Tensor, loss: Tensor) -> tuple[float, int, int]:
    examples = targets.shape[0]
    correct = int((logits.argmax(dim=1) == targets).sum().item())
    return float(loss.detach().item()) * examples, correct, examples


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_loss, batch_correct, batch_examples = _batch_metrics(logits, targets, loss)
        loss_sum += batch_loss
        correct += batch_correct
        examples += batch_examples
    return {"loss": loss_sum / examples, "accuracy": correct / examples}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        batch_loss, batch_correct, batch_examples = _batch_metrics(logits, targets, loss)
        loss_sum += batch_loss
        correct += batch_correct
        examples += batch_examples
    if examples == 0:
        raise ValueError("evaluation loader produced no examples")
    return {"loss": loss_sum / examples, "accuracy": correct / examples, "examples": examples}


def checkpoint_epochs(total_epochs: int, fractions: Sequence[float]) -> tuple[int, ...]:
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("checkpoint fractions must be in (0, 1]")
    return tuple(sorted({max(1, round(total_epochs * fraction)) for fraction in fractions}))


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _reconcile_metrics_for_resume(
    path: Path, checkpoint: dict[str, Any]
) -> None:
    """Make the append-only metric log agree with a recovery checkpoint."""

    checkpoint_epoch = int(checkpoint["epoch"])
    records: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["epoch"]) <= checkpoint_epoch:
                records.append(record)

    if not records or int(records[-1]["epoch"]) < checkpoint_epoch:
        checkpoint_metrics = checkpoint.get("metrics")
        if checkpoint_metrics is None or int(checkpoint_metrics["epoch"]) != checkpoint_epoch:
            raise ValueError("resume checkpoint does not contain matching epoch metrics")
        records.append(checkpoint_metrics)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as metrics_file:
        for record in records:
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def run_dense_training(
    model: CifarResNet,
    fit_loader: DataLoader,
    selection_loader: DataLoader,
    *,
    output: Path,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    epochs: int | None = None,
    stop_after_epoch: int | None = None,
    resume: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Train a dense model with declared diagnostics and rolling recovery state."""

    training = config["training"]
    epochs = int(training["epochs"] if epochs is None else epochs)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["learning_rate"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    save_epochs = checkpoint_epochs(epochs, config["diagnostic"]["checkpoint_fractions"])
    start_epoch = 1
    if resume is not None:
        if int(resume.get("total_epochs", -1)) != epochs:
            raise ValueError("resume checkpoint uses a different total training schedule")
        if int(resume["seed"]) != seed:
            raise ValueError("resume checkpoint seed does not match the requested seed")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        start_epoch = int(resume["epoch"]) + 1
    final_epoch = epochs if stop_after_epoch is None else int(stop_after_epoch)
    if not start_epoch <= final_epoch <= epochs:
        raise ValueError("stop_after_epoch must be within the remaining training schedule")
    if final_epoch < epochs and final_epoch not in save_epochs:
        raise ValueError("partial runs must stop at a declared diagnostic checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    records: list[dict[str, Any]] = []

    if resume is not None:
        _reconcile_metrics_for_resume(metrics_path, resume)
    metrics_mode = "a" if resume is not None else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, final_epoch + 1):
            train_metrics = train_one_epoch(model, fit_loader, optimizer, criterion, device)
            selection_metrics = evaluate(model, selection_loader, criterion, device)
            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "fit": train_metrics,
                "selection": selection_metrics,
            }
            records.append(record)
            scheduler.step()

            checkpoint = {
                "epoch": epoch,
                "total_epochs": epochs,
                "seed": seed,
                "architecture": "resnet18_cifar",
                "num_classes": model.fc.out_features,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": config,
                "metrics": record,
            }
            _save_checkpoint(output / "resume_latest.pt", checkpoint)
            if epoch in save_epochs:
                _save_checkpoint(
                    output / f"checkpoint_epoch_{epoch:03d}.pt",
                    checkpoint,
                )
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_file.flush()
    return records


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was added
        return torch.load(path, map_location="cpu")


def model_from_checkpoint(payload: dict[str, Any]) -> CifarResNet:
    if payload.get("architecture") != "resnet18_cifar":
        raise ValueError("checkpoint is not a CIFAR ResNet-18")
    model = cifar_resnet18(num_classes=int(payload["num_classes"]))
    model.load_state_dict(payload["model_state"])
    return model
