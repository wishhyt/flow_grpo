from collections import defaultdict
import contextlib
import datetime
from concurrent import futures
import hashlib
import json
import os
import time
from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from functools import partial
from ml_collections import config_flags
from torch.utils.data import DataLoader, Dataset, Sampler
import torch
import tqdm
import wandb

import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd14_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd14_ddim_with_logprob import ddim_step_with_logprob


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
flags.DEFINE_string("run_name", None, "Override run name for logging/checkpoints.")

logger = get_logger(__name__)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r", encoding="utf-8") as handle:
            self.prompts = [line.strip() for line in handle.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as handle:
            self.metadatas = [json.loads(line) for line in handle]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k cannot divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def compute_text_embeddings(prompt, text_encoder, tokenizer, device):
    with torch.no_grad():
        prompt_ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        prompt_embeds = text_encoder(prompt_ids)[0]
    return prompt_embeds, prompt_ids


def create_generator(prompts, base_seed, device):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], "big")
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator(device=device).manual_seed(seed)
        generators.append(gen)
    return generators


def compute_log_prob(unet, pipeline, sample, step_index, embeds, config, eta):
    if config.train.cfg:
        noise_pred = unet(
            torch.cat([sample["latents"][:, step_index]] * 2),
            torch.cat([sample["timesteps"][:, step_index]] * 2),
            encoder_hidden_states=embeds,
        ).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
    else:
        noise_pred = unet(
            sample["latents"][:, step_index],
            sample["timesteps"][:, step_index],
            encoder_hidden_states=embeds,
        ).sample

    prev_sample, log_prob = ddim_step_with_logprob(
        pipeline.scheduler,
        noise_pred,
        sample["timesteps"][:, step_index],
        sample["latents"][:, step_index],
        eta=eta,
        prev_sample=sample["next_latents"][:, step_index],
    )
    return prev_sample, log_prob


def eval_loop(
    pipeline,
    test_dataloader,
    text_encoder,
    tokenizer,
    config,
    accelerator,
    global_step,
    reward_fn,
    autocast,
):
    neg_prompt_embed, _ = compute_text_embeddings(
        [""], text_encoder, tokenizer, accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(
        config.sample.test_batch_size, 1, 1
    )

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
        test_dataloader,
        desc="Eval",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, _ = compute_text_embeddings(
            prompts, text_encoder, tokenizer, accelerator.device
        )
        if len(prompt_embeds) < len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[: len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.eval_guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                )
        rewards, _ = reward_fn(images, prompts, prompt_metadata, only_strict=False)
        for key, value in rewards.items():
            rewards_gather = accelerator.gather(
                torch.as_tensor(value, device=accelerator.device)
            ).cpu()
            all_rewards[key].append(rewards_gather)

    if accelerator.is_main_process:
        metrics = {key: torch.cat(value).mean().item() for key, value in all_rewards.items()}
        wandb.log({f"eval_reward_{key}": value for key, value in metrics.items()}, step=global_step)


def unwrap_model(model, accelerator):
    return accelerator.unwrap_model(model)


def save_ckpt(save_dir, pipeline, unet, global_step, accelerator, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    if accelerator.is_main_process:
        os.makedirs(save_root, exist_ok=True)
        if config.use_lora:
            pipeline.unet.save_attn_procs(os.path.join(save_root, "lora"))
        else:
            unwrap_model(unet, accelerator).save_pretrained(os.path.join(save_root, "unet"))


def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.save_dir:
        config.save_dir = os.path.join(config.save_dir, unique_id)

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    eta = getattr(config.sample, "eta", 0.0)
    algorithm = getattr(config.train, "algorithm", "sft")

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
        * num_train_timesteps,
    )
    if FLAGS.run_name:
        config.run_name = FLAGS.run_name
    if accelerator.is_main_process:
        wandb_dir = os.path.join(config.logdir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(name=config.run_name, dir=wandb_dir)
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.safety_checker = None
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]
            else:
                continue
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        pipeline.unet.set_attn_processor(lora_attn_procs)

        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)
        unet = pipeline.unet

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("Please install bitsandbytes for 8-bit Adam") from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    reward_fn = flow_grpo.rewards.multi_score(accelerator.device, config.reward_fn)
    eval_reward_fn = flow_grpo.rewards.multi_score(accelerator.device, config.reward_fn)

    if config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
        collate_fn = GenevalPromptDataset.collate_fn
    elif config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
        collate_fn = TextPromptDataset.collate_fn
    else:
        raise NotImplementedError("Only geneval and general_ocr are supported for SD1.4")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=2,
    )

    neg_prompt_embed, _ = compute_text_embeddings(
        [""], pipeline.text_encoder, pipeline.tokenizer, accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(
        config.sample.train_batch_size, 1, 1
    )
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    unet, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader, test_dataloader
    )

    executor = futures.ThreadPoolExecutor(max_workers=8)

    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        pipeline.unet.eval()
        if epoch % config.eval_freq == 0:
            eval_loop(
                pipeline,
                test_dataloader,
                pipeline.text_encoder,
                pipeline.tokenizer,
                config,
                accelerator,
                global_step,
                eval_reward_fn,
                autocast,
            )
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, pipeline, unet, global_step, accelerator, config)

        samples = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, prompt_ids = compute_text_embeddings(
                prompts, pipeline.text_encoder, pipeline.tokenizer, accelerator.device
            )

            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch * 10000 + i, device=accelerator.device)
            else:
                generator = None

            with autocast():
                with torch.no_grad():
                    images, _, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=eta,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        generator=generator,
                    )

            latents = torch.stack(latents, dim=1)
            log_probs = torch.stack(log_probs, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )

            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, _ = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(
            1, num_train_timesteps
        )
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{key}": value.mean()
                        for key, value in gathered_rewards.items()
                        if "_accuracy" not in key
                    },
                },
                step=global_step,
            )

        if config.per_prompt_stat_tracking:
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, gathered_rewards["avg"], type=algorithm)
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards["avg"] - gathered_rewards["avg"].mean()) / (
                gathered_rewards["avg"].std() + 1e-4
            )

        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[
                accelerator.process_index
            ]
            .to(accelerator.device)
        )

        del samples["rewards"]
        del samples["prompt_ids"]

        mask = samples["advantages"].abs().sum(dim=1) != 0
        remainder = int(mask.sum().item()) % config.train.batch_size
        if remainder != 0:
            false_indices = torch.where(~mask)[0]
            to_change = config.train.batch_size - remainder
            if len(false_indices) >= to_change:
                rand_idx = torch.randperm(len(false_indices))[:to_change]
                mask[false_indices[rand_idx]] = True
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        if total_batch_size % config.train.batch_size != 0:
            raise ValueError("Sample count must be divisible by train.batch_size")

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.unet.train()
            info = defaultdict(list)
            for sample in tqdm(
                samples_batched,
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[: len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        with autocast():
                            _, log_prob = compute_log_prob(
                                unet, pipeline, sample, j, embeds, config, eta
                            )

                        loss = -(sample["advantages"][:, j] * log_prob).mean()
                        info["loss"].append(loss)
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        info = defaultdict(list)
                        global_step += 1

        epoch += 1


if __name__ == "__main__":
    app.run(main)
