#!/bin/bash
#export OMP_NUM_THREADS=8
python \
    -m torch.distributed.launch --nproc_per_node=2 \
     LSP_train-bpe-cn.py \
    --train_input_file ./xinli_qa_data/train_db \
    --eval_input_file ./xinli_qa_data/valid_0_DialogGPT.tsv \
    --output_dir ./output_model/345m-hmwebmix-bpe-v3.2 \
    --model_name_or_path ./pre-train-345m-hmwebmix-bpe-v3.2 \
    --init_checkpoint ./pre-train-345m-hmwebmix-bpe-v3.2/pytorch_model.bin \
    --learning_rate 1e-4  \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --no_token_id False \
    --max_seq_length 512 \
    --gradient_accumulation_steps 4 \
    --valid_step 10000 \
