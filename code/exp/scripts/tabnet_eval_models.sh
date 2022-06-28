MODEL_NAME="ieeecis tabnet_noind tabnet_norelax tabnet_highrelax tabnet_low_na_nd tabnet_lowsteps tabnet_lowbatch"

for model in $MODEL_NAME
do
	echo "Evaluating $model"
	python3 eval.py --dataset ieeecis --model $model --model_path ../models/tabnet/$model.pt --utility_type success_rate
	python3 eval.py --dataset ieeecis --model $model --model_path ../models/tabnet/$model.pt --utility_type average-attack-cost
	echo "Finished evaluating $model"
	echo "-----"
done
