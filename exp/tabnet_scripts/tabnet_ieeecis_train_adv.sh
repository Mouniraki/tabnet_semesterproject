PATH_TO_MODELS="../models/tabnet/"
mkdir $PATH_TO_MODELS
DELAY_VAL=10

for n_shared in `seq 2 3`
do
    for n_ind in `seq 2 3`
    do
            # Name of saved model = <<dataset_name>>_<<n_steps_val>>_<<n_shared_val>>_<<n_ind_val>>-<<n_a_val>>_<<n_d_val>>_<<eps_val>>_<<n_attacks_val>>.pt
            echo "Adversarial training model with n_steps = 4, n_shared = $n_shared and n_ind = $n_ind (no delay):"
            python3 train.py --dataset ieeecis --model tabnet_ieeecis --model_path $PATH_TO_MODELS/ieeecis_4_${n_shared}_${n_ind}_16_16_0_0_nodelay.pt --n_steps 4 --n_shared ${n_shared} --n_ind ${n_ind} --delay 0 --eps 1.0 --eps-sched --utility-type constant --attack_iters 20 --batch_size 2048 --epochs 100
            echo "Adversarial training model with n_steps = 4, n_shared = $n_shared and n_ind = $n_ind (delay = ${DELAY_VAL}):"
            python3 train.py --dataset ieeecis --model tabnet_ieeecis --model_path $PATH_TO_MODELS/ieeecis_4_${n_shared}_${n_ind}_16_16_0_0_delay20.pt --n_steps 4 --n_shared ${n_shared} --n_ind ${n_ind} --delay ${DELAY_VAL} --eps 1.0 --eps-sched --utility-type constant --attack_iters 20 --batch_size 2048 --epochs 100
    done
done