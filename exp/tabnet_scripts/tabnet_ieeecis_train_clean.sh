PATH_TO_MODELS="../models/tabnet/"
mkdir $PATH_TO_MODELS

for n_steps in `seq 2 8`
do
    for n_shared in `seq 1 3`
    do
        for n_ind in `seq 1 3`
        do
            # Name of saved model = <<dataset_name>>_<<n_steps_val>>_<<n_shared_val>>_<<n_ind_val>>-<<n_a_val>>_<<n_d_val>>_<<eps_val>>_<<n_attacks_val>>.pt
            echo "Training model with n_steps = $n_steps, n_shared = $n_shared and n_ind = $n_ind :"
            python3 train.py --dataset ieeecis --model tabnet_ieeecis --model_path $PATH_TO_MODELS/ieeecis_${n_steps}_${n_shared}_${n_ind}_16_16_0_0.pt --n_steps $n_steps --n_shared $n_shared --n_ind $n_ind --epochs 1000 --attack_iters 0
        done
    done
done