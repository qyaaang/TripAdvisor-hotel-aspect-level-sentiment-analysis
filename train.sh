dataset="TripAdvisor_hotel"
model_name="ATAE_BiLSTM"
num_sample=4000
models=("1" "2" "3" "4")
num_epoch=1
optimizers=("SGD" "Adam" "AdaBelief")
learning_rate=0.001
for model in "${models[@]}"; do
  echo "$model"
  if [ "$model" == "1" ]; then
    fracs=(0.35 0.35 0.3)
  elif [ "$model" == "2" ]; then
    fracs=(0.6 0.15 0.25)
  elif [ "$model" == "3" ]; then
    fracs=(0.25 0.6 0.15)
  else
    fracs=(0.25 0.15 0.6)
  fi
  for optimizer in "${optimizers[@]}"; do
    echo "$optimizer"
    python3 train.py --model_name $model_name --dataset $dataset --optimizer "$optimizer" --num_epoch $num_epoch \
                     --num_sample $num_sample --frac_pos "${fracs[0]}" --frac_neu "${fracs[1]}" --frac_neg "${fracs[2]}" \
                     --learning_rate $learning_rate --testset "$model"
    done
done