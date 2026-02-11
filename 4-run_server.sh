# uv run scripts/serve_policy.py --pytorch-device=cpu policy:checkpoint --policy.config=pi0_piper_finetune_put_box --policy.dir=/home/a100_temp/openpi/checkpoints/pi0_piper_finetune_put_box_fir/put_box_fir/50000

# uv run scripts/serve_policy.py --pytorch-device=cpu policy:checkpoint --policy.config=pi0_piper_finetune_put_box_fir --policy.dir=/home/a100_temp/openpi/checkpoints/pi0_piper_finetune_put_box_fir_real/put_box_fir_real/50000

CUDA_VISIBLE_DEVICES=5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_piper_finetune_put_the_yellow_mango_on_the_blue_plate --policy.dir=/home/a100_temp/openpi/checkpoints/pi0_piper_finetune_put_the_yellow_mango_on_the_blue_plate/pi0_piper_finetune_put_the_yellow_mango_on_the_blue_plate_new_model/50000

# CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_piper_finetune_put_the_yellow_mango_on_the_blue_plate_fir --policy.dir=/home/a100_temp/openpi/checkpoints/pi0_piper_finetune_put_the_yellow_mango_on_the_blue_plate/pi0_piper_finetune_put_the_yellow_mango_on_the_blue_plate_model/50000
