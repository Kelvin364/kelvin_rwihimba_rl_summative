### REINFORCE sweep configurations (100k steps, sorted by stochastic reward)

| run_id | hp_baseline | hp_ent_coef | hp_gamma | hp_hidden_sizes | hp_learning_rate | hp_seed | mean_reward | mean_reward_stochastic | success_rate | success_rate_stochastic | mean_final_health | episodes_to_converge |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reinforce_06 | none | 0.01 | 0.99 | [128, 128] | 0.0 | 6 | -16.28 | -1.098 | 0.0 | 0.0 | 0.439 | 1 |
| reinforce_08 | value | 0.01 | 0.99 | [128, 128] | 0.0 | 8 | -16.144 | -1.802 | 0.0 | 0.0 | 0.453 | 1 |
| reinforce_01 | mean | 0.01 | 0.99 | [128, 128] | 0.001 | 1 | -16.137 | -3.252 | 0.0 | 0.0 | 0.453 | 667 |
| reinforce_09 | none | 0.01 | 0.995 | [128, 128] | 0.0 | 9 | -16.394 | -3.359 | 0.0 | 0.0 | 0.449 | 261 |
| reinforce_03 | none | 0.01 | 0.995 | [128, 128] | 0.001 | 3 | -10.517 | -3.388 | 0.0 | 0.0 | 0.448 | 1 |
| reinforce_02 | value | 0.01 | 0.99 | [128, 128] | 0.001 | 2 | -13.313 | -3.77 | 0.0 | 0.0 | 0.449 | 1 |
| reinforce_05 | value | 0.01 | 0.995 | [128, 128] | 0.001 | 5 | -17.55 | -4.227 | 0.0 | 0.0 | 0.45 | 667 |
| reinforce_04 | mean | 0.01 | 0.995 | [128, 128] | 0.001 | 4 | -13.977 | -5.578 | 0.0 | 0.0 | 0.439 | 1 |
| reinforce_07 | mean | 0.01 | 0.99 | [128, 128] | 0.0 | 7 | -21.059 | -7.56 | 0.0 | 0.0 | 0.444 | 667 |
| reinforce_00 | none | 0.01 | 0.99 | [128, 128] | 0.001 | 0 | -16.814 | -16.666 | 0.0 | 0.0 | 0.439 | 1 |
