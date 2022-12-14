for seed in 1 2 3 4 5
  do    
    python cs285/scripts/run_hw3_sac.py \
    --env_name InvertedPendulum-v4 --ep_len 1000 \
    --discount 0.99 --scalar_log_freq 1000 \
    -n 100000 -l 2 -s 256 -b 1000 -eb 2000 \
    -lr 0.0003 --init_temperature 0.1 --exp_name q6a_sac_InvertedPendulum_seed_${seed} \
    --seed $seed
  done