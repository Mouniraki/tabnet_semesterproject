MODEL_NAME="ieeecis tabnet_noind tabnet_norelax tabnet_highrelax tabnet_low_na_nd tabnet_lowsteps tabnet_lowbatch"
  
mkdir ../models/tabnet


for model in $MODEL_NAME
do
        echo "Training $model"
        python3 train.py --dataset ieeecis --model $model --model_path ../models/tabnet/$model.pt --attack_iters 0
        echo "Finished training $model"
        echo "-----"
done
