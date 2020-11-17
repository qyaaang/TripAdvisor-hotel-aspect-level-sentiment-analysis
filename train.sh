dataset="TripAdvisor_hotel"
model_names=("AT_GRU" "ATAE_GRU")
num_sample=4000
trainsets=("1" "2" "3" "4")
testsets=("1" "2" "3" "4")
max_seq_len=80
hidden_dim=30
num_epochs=(100 300 500)
optimizers=("SGD" "Adam" "AdaBelief")
for trainset in "${trainsets[@]}"; do
  if [ "$trainset" == "1" ]; then
    fracs=(0.35 0.35 0.3)
  elif [ "$trainset" == "2" ]; then
    fracs=(0.6 0.15 0.25)
  elif [ "$trainset" == "3" ]; then
    fracs=(0.25 0.6 0.15)
  else
    fracs=(0.25 0.15 0.6)
  fi
  for model_name in "${model_names[@]}"; do
    for num_epoch in "${num_epochs[@]}"; do
      for optimizer in "${optimizers[@]}"; do
        if [ "$optimizer" == "SGD" ]; then
          learning_rate=0.001
          weight_decay=0.001
        else
          learning_rate=0.0001
          weight_decay=0.001
        fi
        printf "\033[1;32mTrainset:\t%s\nModel_name:\t%s\nNum_epoch:\t%s\nOptimizer:\t%s\nLearning_rate:\t%s\nWeight_decay:\t%s\n\033[0m" \
               "$trainset" "$model_name" "$num_epoch" "$optimizer" "$learning_rate" "$weight_decay"
        /usr/bin/python3 train.py --model_name "$model_name" --dataset $dataset \
                                  --learning_rate $learning_rate --weight_decay $weight_decay \
                                  --num_sample $num_sample --frac_pos "${fracs[0]}" --frac_neu "${fracs[1]}" --frac_neg "${fracs[2]}" \
                                  --max_seq_len $max_seq_len --hidden_dim $hidden_dim --num_epoch "$num_epoch" \
                                  --optimizer "$optimizer"
        for testset in "${testsets[@]}"; do
          printf "\033[1;32mTestset:\t%s\nModel_name:\t%s\nOptimizer:\t%s\nLearning_rate:\t%s\nWeight_decay:\t%s\n\033[0m" \
                 "$testset" "$model_name" "$optimizer" "$learning_rate" "$weight_decay"
          /usr/bin/python3 test.py --model_name "$model_name" --dataset $dataset \
                                    --learning_rate $learning_rate --weight_decay $weight_decay \
                                    --num_sample $num_sample --frac_pos "${fracs[0]}" --frac_neu "${fracs[1]}" --frac_neg "${fracs[2]}" \
                                    --max_seq_len $max_seq_len --hidden_dim $hidden_dim --num_epoch "$num_epoch" \
                                    --optimizer "$optimizer" --testset "$testset"
        done
      done
    done
  done
done
