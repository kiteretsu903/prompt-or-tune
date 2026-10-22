# Architecture and code map

The task contract is deliberately small. A BANKING77 query produces one intent.
A support ticket produces category, priority, escalation, and reason code.
Deterministic checks compare those outputs with fixed expected labels.

| Component | Code | Responsibility |
| --- | --- | --- |
| BANKING77 preparation | `banking77_prepare.py`, `banking77.py` | Source hashes, canonical labels, overlap removal, balanced splits, SFT export |
| Prompts | `prompts.py`, `banking77.py`, `banking77_definitions.py` | Versioned policy and intent definitions |
| Output contracts | `schemas.py`, `banking77.py` | Allowed fields and label values |
| Managed inference | `client.py`, `banking77_client.py` | Requests, validation, explicit retries, diagnostics |
| Experiment runners | `run_inference.py`, `banking77_run.py`, `banking77_prompt_experiment.py` | Fixed settings and result serialization |
| Metrics | `metrics.py`, `banking77.py` | Accuracy, confusion counts, latency, and reported tokens |
| Matched comparison guard | `banking77_compare.py` | Reject missing or different evaluation/configuration provenance |
| Public evidence check | `portfolio.py` | Validate complete recorded runs, reproduce metrics/intervals, verify files and splits |

All paths in this table are under `src/fireworks_lab/`. The package name is
retained from the research code so the original tests and imports remain easy
to trace. The installed portfolio command is `prompt-or-tune`.

The offline command uses saved predictions. It does not load `.env`, provision
resources, or call model endpoints. Client and training-export modules remain
available for inspection. The historical prompt runner's deployment names have
been replaced by non-routable `recorded://` aliases; it is not configured for a
new live experiment.

The public replay validates raw responses before using the older support metric
function, whose direct API assumes predictions have already been validated.
The revised BANKING77 evaluator performs its own validation. The separate
legacy report-comparison command intentionally rejects old aggregate-only
reports; the public replay instead binds the retained raw records to exported
inputs and checks the saved expectations explicitly.
