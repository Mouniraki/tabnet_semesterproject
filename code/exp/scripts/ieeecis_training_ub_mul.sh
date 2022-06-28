#UB_MUL_GRID="0.003 0.01 0.03 0.1 0.3"
UB_MUL_GRID="0.3"
mkdir ../models/ieeecis/
for EPS in $UB_MUL_GRID
do
	echo "Training"
	echo $EPS
	python3 train.py --dataset ieeecis --eps $EPS --model_path ../models/ieeecis/ub_mul_$EPS.pt --eps-sched --utility-type multiplicative --attack_iters 10 --batch_size 2048 --epochs 400
done
