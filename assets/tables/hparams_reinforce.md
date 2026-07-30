### REINFORCE sweep configurations (100k steps, sorted by stochastic reward)

| run_id | hp_baseline | hp_ent_coef | hp_episodes_per_batch | hp_gamma | hp_hidden_sizes | hp_learning_rate | hp_max_grad_norm | hp_normalize_advantage | hp_seed | hp_value_epochs | hp_value_lr | mean_reward | mean_reward_stochastic | success_rate | success_rate_stochastic | mean_final_health | episodes_to_converge |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reinforce_07 | value | 0.01 | 4 | 0.99 | [128, 128] | 0.0 | 0.5 | True | 7 | 5 | 0.003 | -22.746 | -6.39 | 0.0 | 0.0 | 0.465 | 31 |
| reinforce_06 | value | 0.01 | 2 | 0.99 | [128, 128] | 0.0 | 0.5 | True | 6 | 5 | 0.003 | -23.861 | -7.23 | 0.0 | 0.0 | 0.465 | 416 |
| reinforce_01 | value | 0.01 | 4 | 0.99 | [128, 128] | 0.001 | 0.5 | True | 1 | 5 | 0.003 | -21.613 | -9.943 | 0.0 | 0.0 | 0.484 | 544 |
| reinforce_09 | mean | 0.01 | 4 | 0.99 | [128, 128] | 0.0 | 0.5 | True | 9 | 5 | 0.003 | -25.804 | -11.069 | 0.0 | 0.0 | 0.479 | 538 |
| reinforce_03 | mean | 0.01 | 4 | 0.99 | [128, 128] | 0.001 | 0.5 | True | 3 | 5 | 0.003 | -33.42 | -11.956 | 0.0 | 0.0 | 0.439 | 1667 |
| reinforce_08 | mean | 0.01 | 2 | 0.99 | [128, 128] | 0.0 | 0.5 | True | 8 | 5 | 0.003 | -28.286 | -12.234 | 0.0 | 0.0 | 0.469 | 2 |
| reinforce_00 | value | 0.01 | 2 | 0.99 | [128, 128] | 0.001 | 0.5 | True | 0 | 5 | 0.003 | -27.645 | -14.05 | 0.0 | 0.0 | 0.453 | 289 |
| reinforce_02 | mean | 0.01 | 2 | 0.99 | [128, 128] | 0.001 | 0.5 | True | 2 | 5 | 0.003 | -28.563 | -14.696 | 0.0 | 0.0 | 0.439 | 1667 |
| reinforce_05 | none | 0.01 | 4 | 0.99 | [128, 128] | 0.001 | 0.5 | True | 5 | 5 | 0.003 | -24.667 | -18.485 | 0.0 | 0.0 | 0.45 | 1667 |
| reinforce_04 | none | 0.01 | 2 | 0.99 | [128, 128] | 0.001 | 0.5 | True | 4 | 5 | 0.003 | -29.611 | -19.452 | 0.0 | 0.0 | 0.454 | 1667 |
