# Portfolio copy

## Project card

**Prompt or Tune**

I compared prompt engineering and LoRA fine-tuning for structured classification.
After an initial support-routing exercise favored a clearer prompt, I moved to
BANKING77 and tested how much of the adapter's advantage remained after adding
label definitions to the prompt. On 385 held-out confirmation queries, those
definitions raised base accuracy from 66.5% to 76.4%; LoRA reached 82.6%. The repository
includes 9,456 recorded predictions, the data splits, and offline verification.
The confirmation queries came from the same public dataset, not production traffic.

## Short description

Prompt engineering versus LoRA for structured classification, with recorded
predictions, matched comparisons, and offline result verification.

## Interview introduction

I started with a support-ticket router. Adding explicit policy rules got 51 of
54 cases right; fine-tuning with that same prompt got 50. Looking at the failures,
the model was confusing lost data with a blocked workflow. Clarifying that rule
in the prompt fixed the remaining base-model errors on the set I had inspected.
The resulting 54/54 score came from debugging cases I had already seen, so it
still needed a fresh test.

BANKING77 gave me a harder classification problem. LoRA improved accuracy on
the official test, but the prompt only listed label names. I then gave both
models explicit definitions and compared all four configurations on the same
unused confirmation queries. The base model improved substantially, while LoRA
still led by 6.2 percentage points.

The evaluation review caught a separate problem: when a retry succeeded, the
client discarded the earlier error. The accuracy scores were still reproducible,
but I could not recover how many HTTP requests had failed. Regression tests now
cover retry logging, malformed outputs, and mismatched comparisons. The fixes
leave the recorded accuracy unchanged; the old retry histories remain unknown.

Qwen supplied the model and Fireworks ran training and serving. The engineering work centered on the data and evaluation pipeline,
including debugging failed runs and making the comparisons reproducible.
I have not deployed it on customer traffic.
