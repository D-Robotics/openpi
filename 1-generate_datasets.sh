# export HF_DATASETS_CACHE=/home/.cache/lerobot

# export XDG_CACHE_HOME=/output
# export HF_LEROBOT_HOME=/home/.cache/lerobot

# uv run examples/piper_real/convert_piper_data_to_lerobot.py --raw-dir /datasett/collect_data/Place_the_oranges_on_the_plate_convert/conversion_completed --repo-id Place_the_oranges_on_the_plate_convert_repo 

# aloha的数据集转换例子


# CUDA_VISIBLE_DEVICES=0 uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /datasets/vla/data_collect/put_the_yellow_mango_on_the_blue_plate --repo-id put_the_yellow_mango_on_the_blue_plate_repo 

CUDA_VISIBLE_DEVICES=0 uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /datasets/vla/data_collect/put_the_yellow_mango_on_the_blue_plate --repo-id put_the_yellow_mango_on_the_blue_plate_repo 


