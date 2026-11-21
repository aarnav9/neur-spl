# Project working preferences

## Long-running experiments

- Before launching a model experiment from Codex, estimate its likely runtime.
- If a run is expected to take more than approximately five minutes, do not launch
  it from Codex unless the user explicitly asks Codex to run it.
- Instead, provide an exact foreground Terminal command for the user to run so the
  experiment does not add to the Codex process's RAM usage.
- Long-run commands should use unbuffered output and print flushed, timestamped
  progress at least once per batch or about once per minute, whichever is sooner.
- Pipe long-run output through `tee` to a named log file and print the result and log
  paths at startup and completion.
- Codex may directly run short smoke tests and validations expected to finish within
  roughly five minutes.

