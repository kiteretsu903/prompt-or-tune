# Results

## BANKING77 official test

Both runs used Qwen3 4B Instruct 2507, the original label-only prompt, reasoning
disabled, temperature zero, and 3,080 official test queries. The recorded result
was 2,097 correct for base and 2,680 for LoRA: 68.0844% versus 87.0130%, a gain
of 18.9286 percentage points. Both runs returned schema-valid outputs with zero
final classification failures.

The original acceptance rule required at least a ten-point gain, at least 70%
tuned accuracy, and at least 99% valid output. The saved scores meet that rule.
The records do not establish absence of this public benchmark from pretraining.

## Stronger-prompt follow-up

Each cell below is a complete run over the indicated split.

| Configuration | Development, 385 | Confirmation, 385 |
| --- | ---: | ---: |
| Base + original | 256 (66.5%) | 256 (66.5%) |
| Base + definitions | 296 (76.9%) | 294 (76.4%) |
| LoRA + original | 330 (85.7%) | 318 (82.6%) |
| LoRA + definitions | 326 (84.7%) | 318 (82.6%) |

The hypothesis was a base-model gain of at least five percentage points while
retaining at least 99% valid output. Both splits met that practical threshold.
All 3,080 follow-up classifications had valid final outputs.

| Paired confirmation comparison | Corrected | Regressed | Net change | Exploratory 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Definitions vs original, base | 45 | 7 | +9.87 points | +7.01 to +12.99 |
| Definitions vs original, LoRA | 15 | 15 | 0.00 points | -2.34 to +2.34 |
| LoRA vs base, original prompt | 76 | 14 | +16.10 points | +12.73 to +19.48 |
| LoRA vs base, definitions prompt | 32 | 8 | +6.23 points | +3.64 to +8.83 |

Intervals use 5,000 paired resamples within intent, seed 20261012. They describe
sampling uncertainty for these fixed runs. They are not adjusted for multiple
comparisons and do not measure variability across training seeds or deployments.
The offline replay reproduces the original resampling order and percentile indices.

## Tokens and latency

| Confirmation configuration | Input tokens | Total tokens | Warm p50 | Warm p95 |
| --- | ---: | ---: | ---: | ---: |
| Base + original | 208,282 | 212,107 | 527.50 ms | 680.65 ms |
| Base + definitions | 852,772 | 856,580 | 520.83 ms | 664.26 ms |
| LoRA + original | 208,282 | 211,724 | 545.50 ms | 723.88 ms |
| LoRA + definitions | 852,772 | 856,199 | 550.02 ms | 686.42 ms |

The longer prompt used about 4.1 times the input tokens. Similar observed latency
in this warm, repeated-prefix workload does not establish a speed advantage.
Serving used on-demand GPU billing; token totals are not a dollar-cost estimate.
Historical latency includes any retry delay within a classification call.

## Earlier support-routing exercise

| Configuration | Exact matches | Accuracy |
| --- | ---: | ---: |
| Base + contract-only prompt | 15/54 | 27.8% |
| Base + explicit policy | 51/54 | 94.4% |
| LoRA + the same explicit policy | 50/54 | 92.6% |
| Base + tightened data-loss boundary | 54/54 | 100.0% |

The last prompt was written after inspecting the three remaining base failures.
The 54 cases therefore serve as development evidence for that change. They are
also variations of a small number of synthetic scenarios, not 54 independent
customer situations. The base explicit-policy run included cold-start delay,
so its latency cannot be compared fairly with deliberately warmed runs.

## What the results support

For the tested support task, improving the policy prompt was the useful next
step. For BANKING77, the existing LoRA adapter retained an advantage after the
base prompt received definitions. The tested adapter with the original prompt
is the supported choice among these configurations: it leads on development,
ties the best confirmation score, and uses the shorter prompt.

This does not prove that a different prompt, examples, retrieval, or a different
model could not match it. The experiment did not test those alternatives.
