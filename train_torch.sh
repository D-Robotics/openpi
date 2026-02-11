# # 建议创建一个新的 train_fixed.sh 文件，内容如下：
# export XDG_CACHE_HOME=/output
# # 关键：禁用内存预分配，并设置一个较低的内存分数
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
# # 注意：这里只设置一次 MEM_FRACTION，避免覆盖

# JAX_TRACEBACK_FILTERING=off 

# 单GPU测试模式
# export CUDA_VISIBLE_DEVICES=1,2,3,5
# export TORCH_DISTRIBUTED_DEBUG=INFO
# uv run python scripts/train_pytorch.py pi0_piper_lora_finetune_put_the_yellow_mango_on_the_blue_plate --exp_name pi0_piper_lora_finetune_single_gpu_test 


export CUDA_VISIBLE_DEVICES=0,1,2,3,4
uv run torchrun --standalone --nnodes=1 --nproc_per_node=5 scripts/train_pytorch.py pi0_piper_lora_finetune_put_the_yellow_mango_on_the_blue_plate --exp_name pi0_piper_lora_finetune_put_the_yellow_mango_on_the_blue_plate_model

# export CUDA_VISIBLE_DEVICES=1,2,3,5
# uv run scripts/train_pytorch.py pi0_piper_lora_finetune_put_the_yellow_mango_on_the_blue_plate --exp_name pi0_piper_lora_finetune_put_the_yellow_mango_on_the_blue_plate_model 