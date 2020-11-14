dataset="TripAdvisor_hotel"
model_name="AT_BiLSTM"
num_sample=4000
#model=(0.35 0.35 0.3)
#testset="1"
#model=(0.6 0.15 0.25)
#testset="2"
#model=(0.25 0.6 0.15)
#testset="3"
model=(0.25 0.15 0.6)
testset="4"
num_epoch=1000
optimizers=("SGD" "Adam" "AdaBelief")
for optimizer in "${optimizers[@]}"; do
  echo "$optimizer"
  python3 train.py --model_name $model_name --dataset $dataset --optimizer "$optimizer" --num_epoch $num_epoch \
                   --num_sample $num_sample --frac_pos "${model[0]}" --frac_neu "${model[1]}" --frac_neg "${model[2]}" \
                   --testset $testset
done
