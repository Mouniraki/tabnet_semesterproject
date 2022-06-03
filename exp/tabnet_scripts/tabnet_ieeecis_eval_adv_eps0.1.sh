UTILITY_TYPES="cost-restrictred average-attack-cost success_rate"
PATH_TO_MODELS="../models/tabnet/"
mkdir $PATH_TO_MODELS

EPS_VAL=0.1
ATTACK_ITERS=20

echo "Model format: [n_steps|n_shared|n_ind|eps_val|n_attack_iters]"
echo "accuracy,cost-restricted,average-attack-cost,success_rate"
for n_steps in `seq 2 8`
do
    for n_shared in `seq 1 3`
    do
        for n_ind in `seq 1 3`
        do
            # Name of saved model = <<dataset_name>>_<<n_steps_val>>_<<n_shared_val>>_<<n_ind_val>>-<<n_a_val>>_<<n_d_val>>_<<eps_val>>_<<n_attacks_val>>.pt
            echo "[${n_steps}|${n_shared}|${n_ind}|${EPS_VAL}|${ATTACK_ITERS}]"
            for ut in $UTILITY_TYPES
            do
                if [ "$ut" = 'cost-restrictred' ]
                then
                    python3 eval.py --dataset ieeecis --model tabnet_ieeecis --model_path $PATH_TO_MODELS/ieeecis_${n_steps}_${n_shared}_${n_ind}_16_16_${EPS_VAL}_${ATTACK_ITERS}.pt --n_steps $n_steps --n_shared $n_shared --n_ind $n_ind --utility_type $ut --force
                else
                    python3 eval.py --dataset ieeecis --model tabnet_ieeecis --model_path $PATH_TO_MODELS/ieeecis_${n_steps}_${n_shared}_${n_ind}_16_16_${EPS_VAL}_${ATTACK_ITERS}.pt --n_steps $n_steps --n_shared $n_shared --n_ind $n_ind --utility_type $ut
                fi
            done
            echo "----------------------------------"
        done
    done
done