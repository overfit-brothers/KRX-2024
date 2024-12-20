python3 scripts/train_DPO.py \
    --model_name "overfit-brothers/dddd" \
    --tokenizer_name "overfit-brothers/dddd" \
    --hf_upload_name "overfit-brothers/dddddpo" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --max_seq_len 1024 \
    --warmup_ratio 0.01 \
    --max_steps 300 \
    --learning_rate 5e-7 \
    --output_dir "outputs" \
    --DPO_mode "offline" \
    --data_config_path "configs/dpo_test.yaml"