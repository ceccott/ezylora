{
  "LoRA_type": "Standard",
  "adaptive_noise_scale": 0,
  "additional_parameters": "",
  "block_alphas": "",
  "block_dims": "",
  "block_lr_zero_threshold": "",
  "bucket_no_upscale": true,
  "bucket_reso_steps": 64,
  "cache_latents": true,
  "cache_latents_to_disk": false,
  "caption_dropout_every_n_epochs": 0.0,
  "caption_dropout_rate": 0,
  "caption_extension": ".txt",
  "clip_skip": 2,
  "color_aug": false,
  "conv_alpha": 1,
  "conv_block_alphas": "",
  "conv_block_dims": "",
  "conv_dim": 1,
  "decompose_both": false,
  "dim_from_weights": false,
  "down_lr_weight": "",
  "enable_bucket": true,
  "epoch": 1,
  "factor": -1,
  "flip_aug": false,
  "full_bf16": false,
  "full_fp16": false,
  "gradient_accumulation_steps": 1,
  "gradient_checkpointing": true,
  "keep_tokens": "0",
  "learning_rate": 0.0003,
  "logging_dir": "{{lora_path['log']}}",
  "lora_network_weights": "",
  "lr_scheduler": "constant",
  "lr_scheduler_args": "",
  "lr_scheduler_num_cycles": "",
  "lr_scheduler_power": "",
  "lr_warmup": 0,
  "max_bucket_reso": 2038,
  "max_data_loader_n_workers": "1",
  "max_resolution": "1024,1024",
  "max_timestep": 1000,
  "max_token_length": "75",
  "max_train_epochs": "",
  "max_train_steps": "",
  "mem_eff_attn": false,
  "mid_lr_weight": "",
  "min_bucket_reso": 256,
  "min_snr_gamma": 0,
  "min_timestep": 0,
  "mixed_precision": "bf16",
  "model_list": "stabilityai/stable-diffusion-xl-base-1.0",
  "module_dropout": 0,
  "multires_noise_discount": 0,
  "multires_noise_iterations": 0,
  "network_alpha": 1,
  "network_dim": 64,
  "network_dropout": 0,
  "no_token_padding": false,
  "noise_offset": 0,
  "noise_offset_type": "Original",
  "num_cpu_threads_per_process": 2,
  "optimizer": "Adafactor",
  "optimizer_args": "",
  "output_dir": "{{lora_path['model']}}",
  "output_name": "{{model_name}}",
  "persistent_data_loader_workers": false,
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
  "prior_loss_weight": 1.0,
  "random_crop": false,
  "rank_dropout": 0,
  "reg_data_dir": "",
  "resume": "",
  "sample_every_n_epochs": 0,
  "sample_every_n_steps": 0,
  "sample_prompts": "",
  "sample_sampler": "euler_a",
  "save_every_n_epochs": 1,
  "save_every_n_steps": 0,
  "save_last_n_steps": 0,
  "save_last_n_steps_state": 0,
  "save_model_as": "safetensors",
  "save_precision": "bf16",
  "save_state": false,
  "scale_v_pred_loss_like_noise_pred": false,
  "scale_weight_norms": 0,
  "sdxl": false,
  "sdxl_cache_text_encoder_outputs": false,
  "sdxl_no_half_vae": true,
  "seed": "1234",
  "shuffle_caption": false,
  "stop_text_encoder_training": 0,
  "text_encoder_lr": 0.0003,
  "train_batch_size": 1,
  "train_data_dir": "{{lora_path['image']}}",
  "train_on_input": true,
  "training_comment": "",
  "unet_lr": 0.0003,
  "unit": 1,
  "up_lr_weight": "",
  "use_cp": false,
  "use_wandb": false,
  "v2": false,
  "v_parameterization": false,
  "v_pred_like_loss": 0,
  "vae_batch_size": 0,
  "wandb_api_key": "",
  "weighted_captions": false,
  "xformers": "xformers"
}