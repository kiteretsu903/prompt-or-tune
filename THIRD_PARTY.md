# Third-party notices

## BANKING77

BANKING77 is a dataset by PolyAI, distributed under the Creative Commons
Attribution 4.0 International license.

- **Source:** [PolyAI task-specific datasets](https://github.com/PolyAI-LDN/task-specific-datasets).
- **Paper:** Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić, [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807), 2020.
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the [publisher's license text](LICENSES/BANKING77-CC-BY-4.0.txt) is included.

The source CSVs are included unchanged. Derived files trim outer whitespace,
canonicalize label spellings, remove exact training/test overlaps and duplicate
training text, and select deterministic splits. The source hashes and split
metadata are included. BANKING77 query text in evaluation records and examples
retains this attribution and license. Its license is not replaced by the
project's MIT code license. No endorsement by PolyAI is implied.

## Models and services

Experiments used Qwen models and Fireworks-managed training/inference. Model
weights and provider services are not included in this repository. Their own
licenses and terms apply to any separate use.

## Python dependencies

The runtime uses Pydantic, the OpenAI Python client, and python-dotenv. The lockfile
records exact dependency versions; each package retains its respective license.
