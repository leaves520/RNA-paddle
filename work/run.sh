CUDA_VISIBLE_DEVICES=0 python main.py \
--n_gram=3 \
--lr=0.0005 \
--save_dir=inference_checkpoint/test \
--dropout_rate=0.2 \
--patience=10 \
--epochs=50 \
--d_model=256 \
--hidden_inter=768 \
--batch_size=32 \
--num_layers=1 \
--n_heads=2 \
--seq_emb=mydata/seq_cb_ns_256_human_3gram