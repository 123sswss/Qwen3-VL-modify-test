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
- The user must personally execute every operation that uses, initializes, reserves, inspects, or otherwise involves a GPU. This includes training, evaluation, inference, model forward/backward passes, CUDA diagnostics, and GPU process or status commands. The assistant must never launch these operations; it may only prepare the implementation and provide the exact command for the user to run. An authorization to implement or run an experiment does not override this execution boundary unless the user explicitly updates this rule in `AGENTS.md`.
- The assistant may execute operations that are guaranteed not to use a GPU, including CPU-only file inspection, source editing, syntax checks, Git operations, log reading, result extraction, and offline statistics over existing artifacts.
- Add every completed experiment to both records: keep the full exact record in `EXPERIMENT_RESULTS.md` and the concise conclusion in `result.md`.
- Add, remove, or reschedule planned experiments in `plan.md`.

## Single-task exception: 2026-10-22 paper evidence handoff

- The user explicitly authorizes the assistant to load the final-LR V1 seed44 epoch5 model and export the six fixed PathVQA Test case maps for `server_handoff_20261022.md`. This exception covers only the required case prefill/inference and ends when this handoff is completed. No training, checkpoint updates, new seeds, or full-dataset inference are authorized. The usual user-only GPU rule remains in force for all other tasks.
