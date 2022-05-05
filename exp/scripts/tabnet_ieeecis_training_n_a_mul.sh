PATH = "../models/tabnet/tabnet_n_a"
mkdir $PATH
for i in `seq 8 512`
do
    python3 train.py --dataset ieeecis --model tabnet_ieeecis --model_path $PATH/$i.pt --n_a $i --epochs 200 --attack_iters 0
    python3 eval.py --dataset ieeecis --model tabnet_ieeecis --model_path $PATH/$i.pt --n_a $i --utility_type average-attack-cost
done