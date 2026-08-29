# Pulled Data And External Dependencies

This directory is the explicit boundary for large or third-party inputs.
Payloads are local and ignored by Git; manifests and retrieval instructions are
tracked.

`manifest.yaml` declares the datasets and upstream research repositories.
Payload directories are created only when they are used:

```text
raw/        Downloaded datasets
processed/  Deterministic splits or derived data
external/   Pinned upstream repositories
```

## Initial Data

The pilot uses CIFAR-100. CIFAR-10 is pulled only for instrumentation debugging.
The canonical dataset is the University of Toronto distribution. The current
pilot fetches its `uoft-cs/cifar100` Hugging Face mirror because the canonical
host can be very slow; the benchmark content and declared split are unchanged.

```bash
make list-data
.venv/bin/pip install -e 'code[hf]'
make pull-pilot PYTHON=.venv/bin/python
```

ImageNet is declared for the scale phase but cannot be fetched automatically.
Place a licensed copy under `pull/raw/imagenet/` only after the method is locked
for the scale experiment.

## External Code

Upstream repositories are not cloned until a baseline is actually scheduled.
Before the first official comparison:

1. resolve and record an immutable commit in `manifest.yaml`;
2. clone into `pull/external/<name>`;
3. record the checkout digest and local modifications;
4. run the released implementation without silent local substitution; and
5. retain failures as baseline outcomes.

Entries marked `PIN_BEFORE_USE` are discovery records, not reproducible locks.
