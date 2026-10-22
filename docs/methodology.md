# Methodology

## Data before model changes

The BANKING77 source files contain 10,003 training and 3,080 test queries. The
preparation code checks their SHA-256 hashes, canonicalizes the publisher's label
spellings, and normalizes whitespace and case for exact-text overlap checks.
Seven training rows overlapped test text and four further training rows were
duplicates. They were removed before sampling; the official test was retained.

With seed 42, the pipeline selected 25 examples per intent for SFT and five per
intent for development. This produced 1,925 training, 385 development, and 7,682
reserve records. SFT serialization uses only the selected training rows. The
public verifier reconstructs these files from the included source CSVs.

Exact-text separation does not rule out semantic similarity or pretraining
contamination. The benchmark's task and label conventions remain shared across
all splits.

## Original LoRA comparison

The recorded configuration used Qwen3 4B Instruct 2507 and rank-8 LoRA for two
epochs, with batch size 32, learning rate 0.0001, a constant schedule, and zero
warmup. There was no training-time validation dataset. Development evaluation
was external. Both inference arms used the same label-only prompt, strict JSON
schema, temperature zero, reasoning disabled, and max_tokens 40.

The research report records that these choices were frozen before final
quality analysis of the official test. One official-test query was also used
as an operational warm-up before the complete base run. Those warm-up records
are retained. The public export was assembled later; its manifest is an integrity
record, not an independently timestamped preregistration.

## Prompt follow-up

The definitions prompt uses the label list and four examples per intent from
the selected SFT training split. The 308
supporting rows are included in
`experiments/followup/definition_training_evidence.json`. The prompt contains
label descriptions and boundary guidance, not demonstrations.

The follow-up held the base model, existing adapter, output contract, generation
settings, and evaluation inputs constant within each comparison. The four arms
were base/original, base/definitions, LoRA/original, and LoRA/definitions. Each ran
on development and then on confirmation. No model was retrained, and the prompt
was not revised after development scores.

Confirmation sampled five examples per intent from the unused reserve, with
seed 20261012. It was excluded from training, development, official test, and
prompt construction by normalized-text checks. Earlier official-test results
were already known, so this is a fresh confirmation sample within the dataset,
not another blind official-test evaluation.

Both saved October deployment snapshots show one B200 GPU, BF16, the same shape
version, and zero-to-one autoscaling. Prompts alternated order in 32-example
batches at concurrency eight per model. Warm-up used a selected training example
and was recorded separately. The resource aliases in this export are not usable
provider endpoints.

## Scoring and review

BANKING77 accuracy is the fraction of queries with the expected intent.
Support-routing exact match requires all four fields to match. Invalid or failed
outputs count as incorrect. The portfolio replay checks raw JSON against the
schema and recorded prediction before recomputing metrics, and requires exact
input text, labels, IDs, model aliases, and available prompt fingerprints.

The historical client recorded the final result of each classification, but
lost failed attempts followed by a successful retry. Zero final failures cannot
establish zero HTTP errors. The later client fix disables SDK retries and records
one explicit retry loop. New comparison reports also carry provenance checks;
historical runs were not retroactively given that metadata.

See [reproduction](reproduction.md) for what can be checked offline and what
would require fresh training or inference.
