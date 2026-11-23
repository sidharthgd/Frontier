"""How often the pure-PPO policy's first-ranked request at a decide call
(decisions18/queue.csv) is the one plain SJF or cache-protecting SJF would
put first (smallest uncached prefill; or, when any waiting request has idle
cached blocks, the one with the most of them).

    .venv/bin/python rl_runs/chameleon/head_agreement.py >> rl_runs/chameleon/decisions18/tables.txt
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
q = pd.read_csv(os.path.join(ROOT, "decisions18", "queue.csv"),
                usecols=["workload", "variant", "seed", "decision", "req", "level", "arrived",
                         "uncached", "idle_blocks", "admitted"])
q = q[q.variant == "base"]
print("\n==== 8. PPO's first-ranked request at decide calls vs the first request of two rules")
print("rule SJF: smallest uncached prefill; rule CP-SJF: most idle cached blocks if any request has some, else SJF")
for w in ["tracelab_c10", "freeinference_c10", "tracelab_c8", "freeinference_c8"]:
    x = q[q.workload == w]
    key = ["seed", "decision"]
    ppo = x.sort_values(key + ["level", "arrived", "req"]).groupby(key).req.first()
    sjf = x.sort_values(key + ["uncached", "arrived", "req"]).groupby(key).req.first()
    idle = x.sort_values(key + ["idle_blocks", "arrived", "req"], ascending=[True, True, False, True, True])
    cp_idle = idle.groupby(key).first()
    cp = sjf.where(cp_idle.idle_blocks == 0, cp_idle.req)
    any_idle = cp_idle.idle_blocks > 0
    admitted = x.groupby(key).admitted.any()
    print(f"{w:18s} calls {len(ppo):7d} | PPO head = SJF head {100 * (ppo == sjf).mean():5.1f}% | "
          f"= CP-SJF head {100 * (ppo == cp).mean():5.1f}% (with an idle prefix waiting: {100 * any_idle.mean():.0f}% of calls, "
          f"agreement there {100 * (ppo == cp)[any_idle].mean():.1f}%; without: {100 * (ppo == cp)[~any_idle].mean():.1f}%)")
