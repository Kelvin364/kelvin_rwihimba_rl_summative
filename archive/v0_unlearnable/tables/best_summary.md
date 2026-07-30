### Best agent per algorithm (400k-step finals)

Reference: random policy -5.13, oracle 17.0 (eval seeds 9000-9019).

| algo | best_run | det_reward | stoch_reward | success_det | success_stoch | final_health | heldout_reward | episodes_to_converge |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | ppo_best | -6.45 | 0.32 | 0.0 | 0.05 | 0.489 | -7.57 | 828 |
| A2C | a2c_best | -9.73 | -6.46 | 0.0 | 0.0 | 0.506 | -10.23 | 535 |
| DQN | dqn_best | -8.46 | -7.88 | 0.0 | 0.0 | 0.455 | -8.21 | 4 |
| REINFORCE | reinforce_best | -16.98 | -16.86 | 0.0 | 0.0 | 0.439 | -16.99 | 1 |
