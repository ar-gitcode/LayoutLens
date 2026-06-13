# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules, unfreeze_last_n_blocks, print_trainable_param_summary
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers, log_optimizer_param_groups
from train_utils import wandb_logger
from train_utils import wandb_clean_logger


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        exp_name: str = "",
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        self.exp_name = exp_name
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)
            if self.rank == 0:
                optim_log_path = os.path.join(
                    self.logging_conf.log_dir, "optimizer_param_groups.txt"
                )
                log_optimizer_param_groups(
                    self.optims, self.model, log_file=optim_log_path
                )

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:   
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # Warm-start fine-tuning: when ``checkpoint.init_weights_only`` is set, the
        # checkpoint is treated purely as a model-weight initialiser. This is the
        # correct mode when the current architecture differs from the one the
        # checkpoint was saved with (e.g. adding an OCA block on top of an
        # e3a/e3b backbone). Restoring the saved optimizer would raise, the saved
        # optimizer has a different param-group layout (no OCA group), and the
        # saved step counters would drop a *fresh* fine-tune into the middle of
        # the LR schedule. Instead we keep the fresh optimizer/scaler built in
        # __init__ and the epoch=0 / steps=0 state set in _setup_components().
        # Missing keys (expected: the new OCA params) and unexpected keys (should
        # be none) were logged above for inspection. Default is False, so all
        # existing configs keep the exact resume behaviour below.
        if bool(getattr(self.checkpoint_conf, "init_weights_only", False)):
            if self.rank == 0:
                logging.info(
                    "init_weights_only=True: warm-start fine-tuning, keeping fresh "
                    "optimizer/scheduler and starting from epoch=0, steps=0 "
                    "(optimizer/scaler/step/epoch state in the checkpoint is ignored)."
                )
            return

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint:
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            self.optims.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load training progress
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Selectively unfreeze last N backbone blocks (room-envelope last2/last4 configs)
        n_unfreeze = getattr(self.optim_conf, "unfreeze_last_n_backbone_blocks", 0)
        if n_unfreeze and n_unfreeze > 0:
            unfreeze_last_n_blocks(self.model, "aggregator", n_unfreeze)
            logging.info(f"Unfroze last {n_unfreeze} aggregator blocks")

        # Print trainable parameter summary on rank 0, and also persist it to
        # logs/${exp_name}/trainable_params.txt so it survives stdout truncation.
        if self.rank == 0:
            param_summary_path = os.path.join(
                self.logging_conf.log_dir, "trainable_params.txt"
            )
            print_trainable_param_summary(self.model, log_file=param_summary_path)

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        # ----- Best-checkpoint selection state -----
        # Rank-0-only running best on Metrics/val/AbsRel_occluded (lower is
        # better). Updated inside run_val(); consumed by save_checkpoint().
        # Also keep ``logs/${exp_name}/ckpts/best.pt`` and ``last.pt`` symlinks.
        self.best_metric_name = "AbsRel_occluded"
        self.best_metric_value = float("inf")  # lower is better
        self.best_metric_epoch = -1
        self.last_val_metrics: Dict[str, float] = {}

        # Initialize wandb (no-op when use_wandb is False or rank != 0)
        if getattr(self.logging_conf, "use_wandb", False):
            extra_cfg: Dict[str, Any] = {
                "max_epochs": self.max_epochs,
                "accum_steps": self.accum_steps,
                "exp_name": self.exp_name,
            }
            try:
                from omegaconf import OmegaConf
                extra_cfg["model"] = OmegaConf.to_container(self.model_conf, resolve=True)
                if self.loss_conf is not None:
                    extra_cfg["loss"] = OmegaConf.to_container(self.loss_conf, resolve=True)
            except Exception:
                pass
            wandb_logger.init_wandb(self.logging_conf, self.exp_name, extra_cfg)

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            # Always write a "last" pointer (and the legacy "checkpoint" name
            # so existing tooling keeps working). Per-epoch numbered copies
            # are still written on the configured save_freq.
            checkpoint_names = ["checkpoint", "last"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )




    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        try:
            if self.mode == "train":
                self.run_train()
                # Optionally run a final validation after all training is done
                self.run_val()
            elif self.mode == "val":
                self.run_val()
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        finally:
            if getattr(self.logging_conf, "use_wandb", False):
                wandb_logger.finish_wandb()

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        # Fresh per-batch metric accumulator; populated inside _step()
        self._reset_val_metric_accum()

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)

        # Aggregate AbsRel/IoU/Normal metrics, log to W&B/TB, update best ckpt.
        if self.rank == 0:
            self._finalize_val_metrics()
            if self._maybe_update_best_metric() and self.mode == "train":
                # Save a pinned best.pt right after a new best is observed so
                # the file matches the metric just printed. The epoch loop
                # also writes last.pt via save_checkpoint().
                self.save_checkpoint(self.epoch, checkpoint_names=["best"])

        del dataloader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict = self._step(
                        batch, self.model, phase, loss_meters
                    )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)


        return True

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                wandb_optim: Dict[str, Any] = {}
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                            wandb_optim[f"optim/{optim_prefix}{option}"] = param_group[option]
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )
                if getattr(self.logging_conf, "use_wandb", False):
                    wandb_optim["optim/where"] = self.where
                    wandb_logger.log_scalars(wandb_optim, self.steps[phase])

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

                if getattr(self.logging_conf, "use_wandb", False):
                    wandb_logger.log_scalars(
                        {f"grad/{k}": v for k, v in grad_norm_dict.items()},
                        self.steps[phase],
                    )

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "layout_depths", "layout_depth_masks",
            "seg_masks",
            "extrinsics", "intrinsics",
            "cam_points", "world_points", "point_masks",
        ]
        string_keys = ["seq_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        
        return batch

    def _process_batch(self, batch: Mapping):
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)

        # Normalize camera extrinsics and points. The function returns new tensors.
        # layout_depths (if present) are normalised with the same scale factor.
        layout_depths_in = batch.get("layout_depths", None)
        (
            normalized_extrinsics,
            normalized_cam_points,
            normalized_world_points,
            normalized_depths,
            normalized_layout_depths,
        ) = normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch["cam_points"],
            world_points=batch["world_points"],
            depths=batch["depths"],
            point_masks=batch["point_masks"],
            layout_depths=layout_depths_in,
        )

        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"]    = normalized_extrinsics
        batch["cam_points"]    = normalized_cam_points
        batch["world_points"]  = normalized_world_points
        batch["depths"]        = normalized_depths
        if normalized_layout_depths is not None:
            batch["layout_depths"] = normalized_layout_depths

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.

        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass. When OCA is enabled, route GT cameras through to the
        # model so it can compute the epipolar bias. When OCA is disabled
        # this branch is skipped and behaviour is identical to E1-E7.
        unwrapped = model.module if hasattr(model, "module") else model
        oca_enabled = getattr(unwrapped, "oca", None) is not None
        if oca_enabled:
            y_hat = model(
                images=batch["images"],
                intrinsics=batch.get("intrinsics"),
                extrinsics=batch.get("extrinsics"),
            )
        else:
            y_hat = model(images=batch["images"])

        # Loss computation
        loss_dict = self.loss(y_hat, batch)

        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])
        self._log_wandb_visuals(log_data, phase, self.steps[phase])

        # Accumulate task metrics during validation (rank-0 only).
        # Metrics are aggregated across batches in `_finalize_val_metrics()`
        # which runs once at the end of `run_val()`.
        if phase == "val" and self.rank == 0:
            self._accumulate_val_metrics(y_hat, batch)
            self._log_wandb_clean(y_hat, batch, phase="val")
        elif phase == "train" and self.rank == 0:
            self._maybe_log_wandb_clean_train(y_hat, batch)

        self.steps[phase] += 1
        return loss_dict

    # ------------------------------------------------------------------ #
    # Validation task metrics, populated inside _step() when phase=='val'
    # ------------------------------------------------------------------ #

    def _reset_val_metric_accum(self):
        """Clear per-batch metric accumulator at the start of a val run."""
        self._val_metric_accum: List[Dict[str, float]] = []
        # Reset the clean-visual budget at the start of each val run so the
        # cap in ``logging.clean_max_scenes_per_eval`` is honored *per* val.
        self._clean_vis_scenes_logged = 0

    def _accumulate_val_metrics(self, predictions, batch):
        """Compute per-frame task metrics and store them in self._val_metric_accum.

        Reuses the canonical helpers in `training/eval_metrics.py` so the same
        AbsRel / IoU / angular-error definitions are used during training-time
        validation and during offline evaluation (evaluations/src/...).
        """
        try:
            from eval_metrics import (
                compute_depth_metrics_with_splits,
                compute_mask_metrics,
                compute_normal_metrics,
            )
        except Exception as exc:
            logging.warning("eval_metrics import failed; skipping val metrics: %s", exc)
            return

        def _np(x):
            if x is None:
                return None
            if isinstance(x, list):
                try:
                    return np.asarray(x)
                except Exception:
                    return x
            if torch.is_tensor(x):
                return x.detach().float().cpu().numpy()
            return np.asarray(x)

        # Layout depth: (B,S,H,W,1) → squeeze
        pred_ld = predictions.get("layout_depth")
        if pred_ld is None:
            return
        pred_ld_np = _np(pred_ld)
        if pred_ld_np.ndim == 5 and pred_ld_np.shape[-1] == 1:
            pred_ld_np = pred_ld_np[..., 0]
        gt_ld_np = _np(batch.get("layout_depths"))
        if gt_ld_np is None:
            return

        valid_np = _np(batch.get("layout_depth_masks"))
        lm_np    = _np(batch.get("layout_masks"))

        # Optional mask predictions
        pred_mask_prob = None
        if "layout_mask_logits" in predictions:
            ml = _np(predictions["layout_mask_logits"])
            if ml.ndim == 5 and ml.shape[2] == 1:
                ml = ml[:, :, 0]
            pred_mask_prob = 1.0 / (1.0 + np.exp(-ml))

        # Optional normal predictions / targets
        pred_n_np = None
        if "layout_normal" in predictions:
            pn = _np(predictions["layout_normal"])
            if pn.ndim == 5 and pn.shape[2] == 3:    # (B,S,3,H,W) → (B,S,H,W,3)
                pn = np.transpose(pn, (0, 1, 3, 4, 2))
            pred_n_np = pn
        gt_n_np = _np(batch.get("layout_normals"))
        gt_nm_np = _np(batch.get("layout_normal_masks"))

        B, S = pred_ld_np.shape[:2]
        for b in range(B):
            for s in range(S):
                rec: Dict[str, float] = {}
                v_s = valid_np[b, s].astype(bool) if valid_np is not None else None
                lm_s = lm_np[b, s] if lm_np is not None else None
                d = compute_depth_metrics_with_splits(
                    pred_ld_np[b, s], gt_ld_np[b, s], v_s, lm_s,
                )
                # Promote to PascalCase + keep ``_subset`` suffixes
                rec["AbsRel_all"]      = d["absrel_all"]
                rec["AbsRel_visible"]  = d["absrel_visible"]
                rec["AbsRel_occluded"] = d["absrel_occluded"]
                rec["RMSE_all"]        = d["rmse_all"]
                rec["Delta1_all"]      = d["delta1_all"]
                rec["Delta1_visible"]  = d["delta1_visible"]
                rec["Delta1_occluded"] = d["delta1_occluded"]
                if pred_mask_prob is not None and lm_np is not None:
                    m = compute_mask_metrics(pred_mask_prob[b, s], lm_np[b, s])
                    rec["Mask_IoU"] = m["iou"]
                    rec["Mask_F1"]  = m["f1"]
                if pred_n_np is not None and gt_n_np is not None:
                    valid_n = gt_nm_np[b, s].astype(bool) if gt_nm_np is not None else None
                    n = compute_normal_metrics(pred_n_np[b, s], gt_n_np[b, s], valid_n)
                    rec["Normal_MeanAngErr"] = n["mean_deg"]
                self._val_metric_accum.append(rec)

    def _finalize_val_metrics(self):
        """Aggregate accumulated val metrics, log to W&B, update best-ckpt state."""
        if not getattr(self, "_val_metric_accum", None):
            return {}
        keys = set()
        for r in self._val_metric_accum:
            keys.update(r.keys())
        agg: Dict[str, float] = {}
        for k in keys:
            vals = np.asarray([r[k] for r in self._val_metric_accum if k in r],
                              dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            agg[k] = float(vals.mean())

        # Log every metric under Metrics/val/<key>. If the config has an
        # explicit ``logging.metric_keys_to_log.val`` list, restrict W&B/TB
        # emission to that allowlist; otherwise emit everything (default).
        allow = None
        mkl = getattr(self.logging_conf, "metric_keys_to_log", None)
        if mkl is not None:
            try:
                val_list = mkl["val"] if "val" in mkl else getattr(mkl, "val", None)
            except Exception:
                val_list = getattr(mkl, "val", None)
            if val_list:
                allow = set(val_list)

        wandb_payload = {f"Metrics/val/{k}": v for k, v in agg.items()}
        if allow is not None:
            wandb_payload = {k: v for k, v in wandb_payload.items() if k in allow}

        if wandb_payload:
            # Val metrics ride the ``val/step`` axis registered in init_wandb,
            # so they're independent of the global monotonic ``_step`` that
            # train logging advances.
            wandb_payload["val/step"] = self.steps["val"]
            try:
                wandb_logger.log_scalars(wandb_payload)
            except Exception as exc:
                logging.debug("wandb metric log failed: %s", exc)
            # TensorBoard does NOT have the monotonic-step constraint, so we
            # write to TB at the actual val step for an accurate val curve.
            for k, v in wandb_payload.items():
                try:
                    self.tb_writer.log(k, v, self.steps["val"])
                except Exception:
                    pass

        # Pretty-print summary line
        nice = "  ".join(f"{k}={v:.4f}" for k, v in sorted(agg.items()))
        logging.info("[val metrics @ epoch %d] %s", self.epoch, nice)

        self.last_val_metrics = agg
        return agg

    def _maybe_update_best_metric(self):
        """Compare current ``last_val_metrics`` against the running best.

        Lower is better for AbsRel_occluded. Returns True iff this epoch is a
        new best.
        """
        m = self.last_val_metrics.get(self.best_metric_name)
        if m is None or not np.isfinite(m):
            return False
        if m < self.best_metric_value:
            self.best_metric_value = float(m)
            self.best_metric_epoch = int(self.epoch)
            logging.info(
                "[best ckpt] new best %s=%.4f at epoch %d",
                self.best_metric_name, m, self.epoch,
            )
            return True
        return False

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard and wandb."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        wandb_metrics: Dict[str, Any] = {}

        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)
                    wandb_metrics[f"{phase}/{key}"] = value

        if wandb_metrics and getattr(self.logging_conf, "use_wandb", False):
            wandb_metrics[f"{phase}/epoch"] = self.epoch
            if phase == "val":
                # Val payloads carry their own ``val/step`` axis (registered in
                # init_wandb via define_metric); they're not subject to the
                # global monotonic ``_step`` rule.
                wandb_metrics["val/step"] = self.steps["val"]
                wandb_logger.log_scalars(wandb_metrics)
            else:
                wandb_logger.log_scalars(wandb_metrics, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )

    def _log_wandb_visuals(self, log_data: Mapping, phase: str, step: int) -> None:
        """Logs RGB + depth visualizations to wandb at the configured interval."""
        if not getattr(self.logging_conf, "use_wandb", False):
            return
        n_steps = getattr(self.logging_conf, "wandb_log_images_every_n_steps", 0)
        if n_steps <= 0 or step % n_steps != 0:
            return
        wandb_logger.log_visual_batch(
            batch=log_data,
            phase=phase,
            step=step,
            epoch=self.epoch,
            max_samples=2,
        )

    # ------------------------------------------------------------------ #
    # Clean qualitative W&B logging, RGB / depth / mask / normal panels +
    # RGB-colored Object3D point clouds. Kept SEPARATE from the legacy
    # overlay/error visuals above so it can be toggled independently.
    # ------------------------------------------------------------------ #

    def _clean_log_kwargs(self) -> Optional[Dict[str, Any]]:
        """Return common kwargs for the clean logger, or None when disabled."""
        if not getattr(self.logging_conf, "use_wandb", False):
            return None
        if not getattr(self.logging_conf, "clean_log_enabled", False):
            return None
        return dict(
            log_2d=bool(getattr(self.logging_conf, "clean_log_2d", True)),
            log_3d=bool(getattr(self.logging_conf, "clean_log_3d", True)),
            log_gt_3d=bool(getattr(self.logging_conf, "clean_log_gt_3d", True)),
            max_points_preview=int(
                getattr(self.logging_conf, "clean_max_points_preview", 50_000)
            ),
            view_indices=list(
                getattr(self.logging_conf, "clean_view_indices", [0]) or [0]
            ),
        )

    def _log_wandb_clean(self, predictions: Mapping, batch: Mapping,
                          phase: str) -> None:
        """Per-val-step clean visual log, honoring per-val scene budget."""
        kwargs = self._clean_log_kwargs()
        if kwargs is None:
            return
        budget = int(getattr(self.logging_conf, "clean_max_scenes_per_eval", 4))
        if budget <= 0:
            return
        already = int(getattr(self, "_clean_vis_scenes_logged", 0))
        if already >= budget:
            return
        # Budget is per-scene, not per-batch: log up to ``remaining`` scenes
        # from this batch in one call.
        remaining = budget - already
        images = batch.get("images")
        if images is not None and hasattr(images, "ndim") and images.ndim >= 5:
            batch_size = int(images.shape[0])
        else:
            batch_size = 1
        n = min(remaining, batch_size)
        try:
            # W&B step must be monotonic across the whole run. The val-local
            # step (``self.steps["val"]``) lags ``self.steps["train"]``, so we
            # stage with ``commit=False``; the staged data rides along on the
            # next train-side commit. This mirrors the val-scalar convention
            # already used by ``_finalize_val_metrics`` above.
            wandb_clean_logger.log_clean_visuals(
                batch=batch,
                predictions=predictions,
                phase=phase,
                wandb_step=self.steps["train"],
                commit=False,
                epoch=self.epoch,
                local_step=self.steps["val"],
                scene_index=already,
                max_samples=n,
                **kwargs,
            )
            self._clean_vis_scenes_logged = already + n
        except Exception as exc:
            logging.debug(f"clean wandb val log failed: {exc}")

    def _maybe_log_wandb_clean_train(self, predictions: Mapping,
                                       batch: Mapping) -> None:
        """Sparse train-time clean visual log gated by ``clean_log_train_every_n_steps``."""
        kwargs = self._clean_log_kwargs()
        if kwargs is None:
            return
        every = int(getattr(self.logging_conf, "clean_log_train_every_n_steps", 0))
        if every <= 0:
            return
        step = self.steps["train"]
        if step % every != 0:
            return
        try:
            wandb_clean_logger.log_clean_visuals(
                batch=batch,
                predictions=predictions,
                phase="train",
                wandb_step=step,
                epoch=self.epoch,
                local_step=step,
                max_samples=1,
                **kwargs,
            )
        except Exception as exc:
            logging.debug(f"clean wandb train log failed: {exc}")


def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data

