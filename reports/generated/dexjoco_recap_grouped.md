# DexJoCo ReCap Grouped Results

|method|label_backend|prompt_mode|eval_prompt_mode|n_eval_rows|mean_success_rate|std_success_rate|pooled_success_rate|binomial_95ci_low|binomial_95ci_high|total_successes|total_episodes|
|---|---|---|---|---|---|---|---|---|---|---|---|
|acp_positive||||5|0.588000|0.039192|0.588000|0.544857|0.631143|294|500|
|baseline||||5|0.684000|0.028705|0.684000|0.643249|0.724751|342|500|
|p2_acp_lora_seed10000|none|acp|acp|3|0.520000|0.069761|0.520000|0.463465|0.576535|156|300|
|p2_base_lora_seed10000|none|base|base|3|0.556667|0.075865|0.556667|0.500451|0.612882|167|300|
|p2_pistar06_value_acp_seed10000|pistar06|indicator|acp|3|0.700000|0.008165|0.700000|0.648143|0.751857|210|300|
|p2_random_positive_seed10000|npz|indicator|acp|3|0.590000|0.021602|0.590000|0.534344|0.645656|177|300|
