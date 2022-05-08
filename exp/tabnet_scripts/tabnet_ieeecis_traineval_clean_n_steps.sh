PATH_TO_MODELS="../models/tabnet/tabnet_n_steps"
mkdir $PATH_TO_MODELS
for i in `seq 2 8`
do
    # Name of saved model = <<dataset_name>>_<<n_steps_val>>_<<n_shared_val>>_<<n_ind_val>>-<<n_a_val>>_<<n_d_val>>_<<eps_val>>_<<n_attacks_val>>.pt
    echo "Training model with n_steps = $i"
    python3 train.py --dataset ieeecis --model tabnet_ieeecis --model_path $PATH_TO_MODELS/ieeecis_$i_2_2_16_16_0_0.pt --n_steps $i --epochs 1000 --attack_iters 0
    echo "Evaluating model"
    python3 eval.py --dataset ieeecis --model tabnet_ieeecis --model_path $PATH_TO_MODELS/ieeecis_$i_2_2_16_16_0_0.pt --n_steps $i --utility_type average-attack-cost
done