# Historical experiment records

These records preserve context for the scored comparison. They are not extra
observations added to the headline accuracy denominators.

| Directory/file | Contents |
| --- | --- |
| `runs/` | Candidate development runs, smoke tests, cold-start records, and the support cold run |
| `execution/` | Failed authentication attempt, successful follow-up warm-up, and start/finish records |
| `followup/` | Historical manifest, runtime versions, prompt-authoring training rows, and deployment snapshots |
| `training_configuration.json` | Settings retained from the research reports, with evidence limits |

The Qwen3 0.6B and Ministral 3B development results belong to other models. They
are not baselines for the final tuned Qwen3 4B comparison. The support cold run
contains five final 503 failures and remains separate from the later complete
warm run. The first October follow-up attempt stopped at authentication; it is
retained alongside the successful attempt.

Resource aliases are non-routable. Request IDs were pseudonymized. Error messages,
model output, latency, and usage remain inspectable. Historical retry attempts
that were never saved cannot be reconstructed from this archive.
