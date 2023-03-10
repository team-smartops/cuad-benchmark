CUDA_VISIBLE_DEVICES=0,1 python train.py \
        --output_dir ./train_models/deberta-v2-xxlarge \
        --model_type deberta \
        --model_name_or_path deberta-v2-xxlarge \
        --train_file ./data/train_separate_questions.json \
        --predict_file ./data/test.json \
        --do_train \
        --version_2_with_negative \
        --learning_rate 3e-6 \
        --num_train_epochs 1 \
        --per_gpu_eval_batch_size=2  \
        --per_gpu_train_batch_size=2 \
        --max_seq_length 512 \
        --max_answer_length 512 \
        --doc_stride 256 \
        --save_steps -1 \
        --n_best_size 20 \
        --overwrite_output_dir