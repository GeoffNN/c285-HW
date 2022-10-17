export lr=.0003
export actor_update_frequency=1

for seed in 5 4 3 2 1
  do
    python cs285/scripts/run_hw3_sac.py \
    --env_name HalfCheetah-v4 --ep_len 150 \
    --discount 0.99 --scalar_log_freq 1500 \
    -n 220000 -l 2 -s 256 -b 1500 -eb 1500 \
    -lr $lr --init_temperature 0.1 --exp_name q6b_sac_HalfCheetah_seed_${seed}_actor_update_frequency_${actor_update_frequency}_lr_${lr} \
    --seed $seed --actor_update_frequency ${actor_update_frequency}
  done