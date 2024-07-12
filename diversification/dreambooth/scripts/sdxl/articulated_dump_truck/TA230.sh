
#!/bin/bash

cd /mnt/ssd2/xin/repo/DART/diversification/dreambooth

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export INSTANCE_DATA_DIR="/mnt/ssd2/xin/repo/DART/diversification/instance_data/articulated_dump_truck/TA230"
export OUTPUT_DIR="/mnt/ssd2/xin/repo/DART/diversification/dreambooth/output/sdxl/articulated_dump_truck/TA230"
export CLASS_DATA_DIR="/mnt/ssd2/xin/repo/DART/diversification/dreambooth/class_data/sdxl/articulated_dump_truck"
export CONFIG_FILE="/home/chenxin/.cache/huggingface/accelerate/default_config.yaml"

accelerate launch --config_file=$CONFIG_FILE\
  train_dreambooth_lora_sdxl_fix-snr.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --instance_data_dir=$INSTANCE_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a <TA230> articulated dump truck" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=0.0001 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1400 \
  --checkpointing_steps=400 \
  --mixed_precision="bf16" \
  --prior_loss_weight=1.0 \
  --num_validation_images=4 \
  --validation_steps=400 \
  --validation_prompt="A photo of a <TA230> articulated dump truck on a construction site. The image is high quality and photorealistic. The <TA230> articulated dump truck may be partially visible, at a distance, or obscured, ensuring a variety of training examples for object detection. The background is complex, providing a realistic context." \
  --with_prior_preservation \
  --class_data_dir=$CLASS_DATA_DIR \
  --class_prompt="a photo of articulated dump truck" \
  --snr_gamma=5.0 \
  --train_text_encoder 