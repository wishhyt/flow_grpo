# 1 GPU
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29511 scripts/train_sd14.py --config config/grpo.py:general_ocr_sd14
