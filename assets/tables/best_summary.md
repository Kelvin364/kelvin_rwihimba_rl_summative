### Best agent per algorithm (400k-step finals)

Reference policies on eval seeds 9000-9019: random -13.88, oracle 14.55. `pct_of_oracle` places each agent on that scale (0% = random, 100% = oracle) and is the only column comparable across reward-function versions.

| algo | best_run | det_reward | stoch_reward | success_det | success_stoch | final_health | heldout_reward | gen_gap | pct_of_oracle | episodes_to_converge |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | ppo_best | 1.73 | 8.68 | 0.6 | 0.65 | 0.794 | 7.09 | 1.83 | 79.4 | 602 |
| DQN | dqn_best | 8.57 | 8.66 | 0.65 | 0.65 | 0.858 | 7.1 | 2.32 | 79.3 | 935 |
| A2C | a2c_best | -4.65 | 1.2 | 0.4 | 0.45 | 0.722 | -0.01 | 4.26 | 53.1 | 585 |
| REINFORCE | reinforce_best | -21.65 | -8.42 | 0.0 | 0.0 | 0.489 | -7.06 | 1.48 | 19.2 | 543 |
