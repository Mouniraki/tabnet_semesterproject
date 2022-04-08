UB_GRID="0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0 50.0 100.0 200.0 500.0"
mkdir ../../models/home_credit/
for EPS in $L1_GRID
do
	echo "Training"
	echo $EPS
	python ../train.py --dataset home_credit --eps $EPS --model_path ../../models/home_credit/l1_$EPS.pt --eps-sched --same-cost --attack_iters 10 --batch_size 2048 --epochs 400
done
