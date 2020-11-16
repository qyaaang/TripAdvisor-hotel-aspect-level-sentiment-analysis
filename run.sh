dataset="TripAdvisor_hotel"
model_name="AT_GRU"
num_sample=4000
model="1"
max_seq_len=80
hidden_dim=30
num_epoch=1000
optimizer="SGD"
learning_rate=0.0001
weight_decay=0.001
if [ "$model" == "1" ]; then
  fracs=(0.35 0.35 0.3)
elif [ "$model" == "2" ]; then
  fracs=(0.6 0.15 0.25)
elif [ "$model" == "3" ]; then
  fracs=(0.25 0.6 0.15)
else
  fracs=(0.25 0.15 0.6)
fi
printf "\033[1;32mModel:\t%s\nModel_name:\t%s\nOptimizer:\t%s\nLearning_rate:\t%s\nWeight_decay:\t%s\n\033[0m" \
               "$model" "$model_name" "$optimizer" "$learning_rate" "$weight_decay"
/usr/bin/python3 train.py --model_name $model_name --dataset $dataset \
                --learning_rate $learning_rate --weight_decay $weight_decay \
                --num_sample $num_sample --frac_pos "${fracs[0]}" --frac_neu "${fracs[1]}" --frac_neg "${fracs[2]}" \
                --max_seq_len $max_seq_len --hidden_dim $hidden_dim --num_epoch $num_epoch \
                --optimizer $optimizer --testset $model --early_stopping
