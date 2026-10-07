# BUGS — what broke, and what caught it

| # | Phase | What broke | What caught it | Fix |
|---|---|---|---|---|
| 1 | 0 | The MPS model rate kept rising from run to run (185 → 233 → 288 fps). One warm-up batch was not enough, so the first runs were timing GPU warm-up rather than steady state. | The timed runs climbed steadily instead of scattering around a mean. | The warm-up is now one full untimed pass over all batches. The later runs scatter at 206–255 fps with no trend. |
