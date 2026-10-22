# Limitations

The experiments measure benchmark classification. They do not establish behavior
on real customer traffic or suitability for a production support system.

The support set contains repeated variants of a small number of synthetic
situations. Its final prompt was written after inspecting failures, so 100%
exact match is a development result. Shared amounts and scenario structure also
make textual train/eval separation a weak generalization test for that task.

BANKING77 is public. Pretraining exposure was not assessed, and exact normalized
text checks do not eliminate semantic near-duplicates. Some queries are ambiguous
or follow label conventions that would need review in an actual application.
Scoring measures agreement with those labels, not an independently adjudicated
customer intent.

The confirmation sample comes from the unused training reserve of the same
dataset. It was excluded from this project's SFT and prompt construction, but
it is not an external-domain test. Five queries per intent are too few for strong
per-intent claims. Only one definitions prompt and one retained adapter were
compared. Few-shot prompts, retrieval, repeated training seeds, and other models
could change the outcome.

Bootstrap intervals are conditional on the sampled queries and fixed model
runs. They do not quantify training or serving variation and are not corrected
for multiple comparisons.

Historical records preserve final errors, output, latency, and usage. They do
not preserve successful-retry histories, so the number of HTTP attempts and
recovered failures is unknown. Token counts include reported usage, not a full
billing audit. The latency measurements are not controlled production benchmarks.

The original report's training settings and approximate charges are retained
as reported metadata. The public bundle does not contain provider billing
receipts, immutable model weights, or an independently timestamped freeze.
File hashes detect changes relative to this release; they do not authenticate
the provider's original responses.

Model weights and live endpoints are not included. The public CLI can verify
saved evidence without paid services. Retraining or regenerating predictions
requires a new account configuration and has not been validated as a turnkey
workflow from this export.
