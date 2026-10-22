# How the project developed

## Start with a policy problem

The first task routed support tickets into category, priority, escalation, and
reason code. The contract-only prompt usually recognized the broad category,
but it did not express the policy well enough to assign the other fields.
Adding explicit rules raised exact match from 15/54 to 51/54.

The remaining failures involved saved work disappearing after reconnection.
LoRA did not fix that boundary and added one priority error, ending at 50/54.
I clarified the distinction between lost data and a blocked workflow in the
prompt. That reached 54/54 on the inspected set. It was useful debugging evidence,
but I could no longer treat the set as an untouched final test.

## Move to a harder classification task

BANKING77 offered 77 adjacent intents and an official test split. I removed
exact train/test text overlap before selecting training examples. The original
prompt listed the labels without examples or definitions.

Two smaller model candidates did not become the final training comparison.
The research notes record that Qwen3 0.6B was not available through the required
managed-training path. A Ministral 3B job stalled while provisioning and was
cancelled before processing examples. Their development runs remain in the
archive. The final Qwen3 4B run improved official-test accuracy from 68.1% to 87.0%.

## Challenge the baseline

The fine-tuning gain raised another question: was the label-only prompt leaving
out conventions that training had to teach? The PIN-related examples assigned
to `get_physical_card` made that concern concrete.

I wrote one definitions prompt from selected training examples and compared it
with the original prompt for both models. The base model gained 9.87 percentage
points on confirmation. LoRA's advantage narrowed from 16.10 to 6.23 points but
remained. The original LoRA prompt was still the practical choice among those
tested because the longer prompt did not improve its confirmation score.

## Review the measurement tools

The subsequent review reproduced the scores and data splits, but found three
tooling gaps. Successful retries discarded previous errors. The aggregate
comparison command could accept different models and datasets. The evaluator
could count malformed non-null prediction objects as valid.

Regression tests reproduced those cases before the fixes. The client now keeps
attempt-level diagnostics, the BANKING77 evaluator validates predictions, and
the comparison command rejects missing or mismatched provenance. Re-evaluating
the saved outputs left the scores unchanged. The historical retry frequency
remains unknown because those records were never saved.

## Prepare a public record

This repository is a later, curated export with a single initial release commit.
The private research checkout is retained separately. The public history starts
at release preparation; it is not presented as a sequence of contemporaneous
experiment commits.

The export adds offline verification, documentation, attribution, and readable
examples. It does not add new model responses or reinterpret a failed run as a
success. Candidate baselines, cold-start failures, and authentication failures
are retained separately from the complete scored runs.
