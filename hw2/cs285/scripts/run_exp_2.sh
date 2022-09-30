for r in .01 .008 .006
    do
    for b in 7000 9000
        do 
        echo r=${r} b=${b}
        python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
        --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $r -rtg \
        --exp_name "q2_b${b}_r${r}" \
        --which_gpu 2
        done
    done

