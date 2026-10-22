# Reproduce the recorded results

## Install and inspect

Python 3.13 is the tested environment. From a checkout of this repository:

```bash
uv sync --locked --extra dev --python 3.13
uv run --locked prompt-or-tune demo
uv run --locked prompt-or-tune replay --output output/replayed-results.json
uv run --locked prompt-or-tune verify
uv run --locked pytest -q
uv run --locked ruff check src tests
```

Installation needs package downloads unless they are cached. The commands after
installation do not need credentials or network access. Tests block outbound
socket connections. The demo reads a recorded confirmation example; it is not
an interactive model demo.

The CLI expects `evaluation/` and `data/` in the working directory. Use
`--root /path/to/checkout` when invoking an installed package from elsewhere.
The wheel contains code, while the repository contains the evidence bundle.

## What verification checks

Replay validates all 14 complete scored runs against the retained inputs: IDs,
text, expected labels, model aliases, schema-valid raw outputs, predictions,
and required diagnostics. It recomputes aggregate and per-intent metrics, paired
corrections/regressions, and all ten stored bootstrap intervals.

`verify` additionally checks the release's exact file inventory and hashes,
reconstructs BANKING77 splits from the source CSVs, checks the SFT export, and
reproduces the confirmation sample. It checks that the 308 prompt-authoring
examples belong to the selected training set. Operational and candidate records
under `experiments/` are hash-checked archival evidence; they are not mixed into
the 9,456 scored classifications.

## Export provenance

`evaluation/export_provenance.json` records private source paths relative to the
research root, source hashes, published paths, hashes, and transformations.
Provider deployment/account resource names are replaced by stable `recorded://`
aliases, response IDs are pseudonymized, and local absolute-path prefixes are
removed. Query text, expected labels, predictions, latency, usage, and error
meaning are retained. The research checkout itself was not changed.

The release manifest describes the derivative export. Historical source-code
hashes in the archived run manifest refer to the code before the subsequent
review fixes. They are not hashes of the current public implementation, and
that archived manifest cannot be used to launch a new run of the revised code.

The expected-results file derives from the saved reports. Its metrics are
checked against raw predictions by the public replay; they are not accepted
merely because they appear in a report. Hash verification is local integrity
checking, not external attestation of provenance or preregistration.

## Fresh inference and retraining

No adapter weights or working private endpoints are distributed. Inspectable
clients, prompts, SFT data, and recorded configuration are included, but a new
experiment would need your own credentials, supported model/deployment IDs,
a newly frozen protocol, and a separate spending decision.

No live inference or training was run to prepare this portfolio export. The
operational effects of the revised retry client are covered by mocked-request
tests, not newly measured production or provider latency.
