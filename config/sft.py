import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 24
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.algorithm = 'sft'
    # Change ref_update_step to a small number, e.g., 40, to switch to OnlineSFT.
    config.train.ref_update_step=10000000
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 100
    config.sample.global_std=True
    config.train.ema=True
    config.save_freq = 40 # epoch
    config.eval_freq = 40
    config.save_dir = 'logs/geneval/sd3.5-M-sft'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd14():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.sample.num_steps = 50
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 5.0
    config.sample.eval_guidance_scale = 5.0
    config.sample.eta = 0.0

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 2
    config.sample.test_batch_size = 4

    config.train.algorithm = "sft"
    config.train.ref_update_step = 10000000
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 1.0
    config.train.beta = 0.0

    config.sample.global_std = True
    config.mixed_precision = "fp16"
    config.save_dir = "logs/ocr/sd1.4-sft"

    config.reward_fn = {
        "ocr": 1.0,
    }
    config.prompt_fn = "general_ocr"
    config.per_prompt_stat_tracking = True
    return config

def geneval_sd14():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.sample.num_steps = 50
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 5.0
    config.sample.eval_guidance_scale = 5.0
    config.sample.eta = 0.0

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 2
    config.sample.test_batch_size = 4

    config.train.algorithm = "sft"
    config.train.ref_update_step = 10000000
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 1.0
    config.train.beta = 0.0

    config.sample.global_std = True
    config.mixed_precision = "fp16"
    config.save_dir = "logs/geneval/sd1.4-sft"

    config.reward_fn = {
        "geneval": 1.0,
    }
    config.prompt_fn = "geneval"
    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    config.resolution = 512
    config.sample.train_batch_size = 24
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    
    config.train.algorithm = 'sft'
    # Change ref_update_step to a small number, e.g., 40, to switch to OnlineSFT.
    config.train.ref_update_step=10000000
    
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 100   
    config.sample.global_std=True
    config.train.ema=True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M-sft'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_4gpu():
    """
    PickScore SD3 SFT 配置 - 4x GPU 优化
    """
    gpu_number = 4
    config = compressibility()
    
    # 模型与数据集路径
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.save_dir = 'logs/pickscore/sd3.5-M-sft-4gpu'
    
    # 采样与分辨率设置
    config.resolution = 512
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.global_std = True
    config.sample.num_image_per_prompt = 8  # 减小组大小以适应显存
    
    # 训练批次设置
    config.sample.train_batch_size = 2
    config.train.batch_size = config.sample.train_batch_size
    config.sample.test_batch_size = 16
    
    # 计算 num_batches_per_epoch
    config.sample.num_batches_per_epoch = int(
        48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    config.train.gradient_accumulation_steps = max(config.sample.num_batches_per_epoch // 2, 1)

    # SFT 特有配置
    config.train.algorithm = 'sft'
    config.train.ref_update_step = 10000000

    # 训练超参数
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 100
    config.train.learning_rate = 1e-4
    config.train.ema = True
    
    # 显存与硬件优化
    config.train.use_8bit_adam = True
    config.activation_checkpointing = True
    config.use_lora = True
    config.mixed_precision = "bf16"

    # 奖励函数与策略
    config.reward_fn = {"pickscore": 1.0}
    config.prompt_fn = "pickscore"
    config.per_prompt_stat_tracking = True

    # 频率控制
    config.save_freq = 10
    config.eval_freq = 60

    return config


def get_config(name):
    return globals()[name]()
