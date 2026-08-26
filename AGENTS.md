# Project Instructions

- Read `EXPERIMENT_RESULTS.md` before planning, implementing, or interpreting an experiment.
- Every completed experiment must be added to `EXPERIMENT_RESULTS.md` in the same task, including failed and negative experiments.
- Record the exact experiment name, dataset, seed, controlled change, Overall score, available breakdowns, diagnostics, and output path.
- Never silently replace or delete historical results. Add an explicit correction note instead.
- Update the ledgers immediately after a completed experiment, but do not create a ledger-only commit or push. Leave ledger changes local and include them with the next related code commit and push.
- If a push fails, report the unpushed commit hash clearly.
- Read and write text files with explicit UTF-8 encoding when using PowerShell.
- Treat the Windows workspace as the only source of truth: edit and test locally, commit and push locally, then update the AutoDL checkout only with a fast-forward pull.
- Before any server-side Git pull, run `source /etc/network_turbo`; the standard sync command is `source /etc/network_turbo; git pull --ff-only`.
- Do not edit source files or start experiments on the server unless the user explicitly authorizes that specific run.
- Add every completed experiment to both records: keep the full exact record in `EXPERIMENT_RESULTS.md` and the concise conclusion in `result.md`.
- Add, remove, or reschedule planned experiments in `plan.md`.
