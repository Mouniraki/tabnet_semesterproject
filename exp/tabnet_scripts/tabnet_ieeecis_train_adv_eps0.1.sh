PATH_TO_MODELS="../models/tabnet/"
mkdir $PATH_TO_MODELS

DELAY_VAL=10
DELAY_THRESHOLD=100
EPS_VAL=0.1
ATTACK_ITERS=20

for n_steps in `seq 2 8`
do
    for n_shared in `seq 1 3`
    do
        for n_ind in `seq 1 3`
        do
            # Name of saved model = <<dataset_name>>_<<n_steps_val>>_<<n_shared_val>>_<<n_ind_val>>-<<n_a_val>>_<<n_d_val>>_<<eps_val>>_<<n_attacks_val>>.pt
            echo "Training model with n_steps = $n_steps, n_shared = $n_shared, n_ind = $n_ind, eps = ${EPS_VAL} and n_attacks = ${ATTACK_ITERS} :"
            python3 train.py --dataset ieeecis --model tabnet_ieeecis --model_path $PATH_TO_MODELS/ieeecis_${n_steps}_${n_shared}_${n_ind}_16_16_${EPS_VAL}_${ATTACK_ITERS}.pt --n_steps $n_steps --n_shared $n_shared --n_ind $n_ind --delay ${DELAY_VAL} --delay_threshold ${DELAY_THRESHOLD} --eps ${EPS_VAL} --eps-sched --utility-type constant --attack_iters ${ATTACK_ITERS} --batch_size 2048 --epochs 400 
        done
    done
done