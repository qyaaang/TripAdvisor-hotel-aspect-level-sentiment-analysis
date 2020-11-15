dataset="TripAdvisor_hotel"
model_names=("AT_GRU" "ATAE_GRU")
num_sample=4000
models=("1" "2" "3" "4")
max_seq_len=80
hidden_dim=30
num_epoch=100
optimizers=("SGD" "Adam" "AdaBelief")
learning_rates=(0.001)
for model in "${models[@]}"; do
  if [ "$model" == "1" ]; then
    fracs=(0.35 0.35 0.3)
  elif [ "$model" == "2" ]; then
    fracs=(0.6 0.15 0.25)
  elif [ "$model" == "3" ]; then
    fracs=(0.25 0.6 0.15)
  else
    fracs=(0.25 0.15 0.6)
  fi
  for model_name in "${model_names[@]}"; do
    for optimizer in "${optimizers[@]}"; do
      for learning_rate in "${learning_rates[@]}"; do
        printf "\033[1;32mModel:\t%s\nModel_name:\t%s\nOptimizer:\t%s\nLearning_rate:\t%s\n\033[0m" \
               "$model" "$model_name" "$optimizer" "$learning_rate"
        python3 train.py --model_name "$model_name" --dataset $dataset \
                         --learning_rate "$learning_rate" \
                         --num_sample $num_sample --frac_pos "${fracs[0]}" --frac_neu "${fracs[1]}" --frac_neg "${fracs[2]}" \
                         --max_seq_len $max_seq_len --hidden_dim $hidden_dim --num_epoch $num_epoch \
                         --optimizer "$optimizer" --testset "$model"
      done
    done
  done
done
