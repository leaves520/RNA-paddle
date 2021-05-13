n_grams=( 1 3 )
num_layers=( 1 2 )
seeds=( 1 2 3 )
for n_gram in ${n_grams[*]}
do
  for num_layer in ${num_layers[*]}
  do
    n_heads=$((num_layer*2))
    for seed in ${seeds[*]}
    do
    echo "$n_gram, $num_layer, $n_heads, $seed"
    CUDA_VISIBLE_DEVICES=0 python main.py \
    --n_gram="$n_gram" \
    --lr=0.0005 \
    --save_dir=inference_checkpoint/transformers_gru/"$n_gram"_gram_256_"$num_layer"_"$n_heads"_gru_seq_seed_"$seed" \
    --dropout_rate=0.2 \
    --patience=7 \
    --epochs=40 \
    --d_model=256 \
    --hidden_inter=768 \
    --batch_size=32 \
    --num_layers="$num_layer" \
    --n_heads=$n_heads \
    --seq_emb=mydata/seq_cb_ns_256_human_"$n_gram"gram
    done
  done
done

