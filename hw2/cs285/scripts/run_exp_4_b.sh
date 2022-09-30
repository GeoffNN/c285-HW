export b=50000
export r=.02

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r \
--exp_name q4_b${b}_lr${r}

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r-rtg \
--exp_name q4_b${b}_lr${r}_rtg

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r --nn_baseline \
--exp_name q4_b${b}_lr${r}_nnbaseline

python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r -rtg --nn_baseline \
--exp_name q4_b${b}_lr${r}_rtg_nnbaseline