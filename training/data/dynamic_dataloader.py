# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Tuple

from hydra.utils import instantiate
import random
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, Sampler
from abc import ABC, abstractmethod

from .worker_fn import get_worker_init_fn

class DynamicTorchDataset(ABC):
    def __init__(
        self,
        dataset: dict,
        common_config: dict,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
        max_img_per_gpu: int = 48,
    ) -> None:
        self.dataset_config = dataset
        self.common_config = common_config
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.max_img_per_gpu = max_img_per_gpu

        # Instantiate the dataset
        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False)

        # Extract aspect ratio and image number ranges from the configuration
        self.aspect_ratio_range = common_config.augs.aspects  # e.g., [0.5, 1.0]
        self.image_num_range = common_config.img_nums    # e.g., [2, 24]

        # Validate the aspect ratio and image number ranges
        if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
            raise ValueError(f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}")
        if len(self.image_num_range) != 2 or self.image_num_range[0] < 1 or self.image_num_range[0] > self.image_num_range[1]:
            raise ValueError(f"image_num_range must be [min, max] with 1 <= min <= max, got {self.image_num_range}")

        # Create samplers
        self.sampler = DynamicDistributedSampler(self.dataset, seed=seed, shuffle=shuffle)
        self.batch_sampler = DynamicBatchSampler(
            self.sampler,
            self.aspect_ratio_range,
            self.image_num_range,
            seed=seed,
            max_img_per_gpu=max_img_per_gpu
        )

    def get_loader(self, epoch):
        print("Building dynamic dataloader with epoch:", epoch)

        # Set the epoch for the sampler
        self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        # Create and return the dataloader
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
        )
        

class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """
    def __init__(self,
                 sampler,
                 aspect_ratio_range,
                 image_num_range,
                 epoch=0,
                 seed=42,
                 max_img_per_gpu=48):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            image_num_range: List containing [min_images, max_images] per sample.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
        """
        self.sampler = sampler
        self.aspect_ratio_range = aspect_ratio_range
        self.image_num_range = image_num_range
        self.rng = random.Random()

        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: 1.0 for num_images in range(image_num_range[0], image_num_range[1]+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if self.image_num_range[0] <= n <= self.image_num_range[1]])

        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Sample random image number and aspect ratio
                random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                random_aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)

                # Update sampler parameters
                self.sampler.update_parameters(
                    aspect_ratio=random_aspect_ratio,
                    image_num=random_image_num
                )

                # Calculate batch size based on max images per GPU and current image number
                batch_size = self.max_img_per_gpu / random_image_num
                batch_size = np.floor(batch_size).astype(int)
                batch_size = max(1, batch_size)  # Ensure batch size is at least 1

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)  # item is (idx, aspect_ratio, image_num)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples

                if not current_batch:
                    break  # No more data to yield

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self):
        # Return a large dummy length
        return 1000000


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self.aspect_ratio = None
        self.image_num = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num, aspect_ratio).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (idx, self.image_num, self.aspect_ratio,)

    def update_parameters(self, aspect_ratio, image_num):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            image_num: The number of images to set.
        """
        self.aspect_ratio = aspect_ratio
        self.image_num = image_num


class ManifestEvalLoader:
    """Deterministic loader for manifest-driven evaluation.

    Drop-in replacement for ``DynamicTorchDataset`` in val configs when the
    inner dataset has a ``manifest_path`` set. Batches are produced in manifest
    order, grouped so every batch contains samples with identical ``num_views``
    (required because tensors are stacked across the batch dim in the trainer).
    """

    def __init__(
        self,
        dataset: dict,
        common_config: dict,
        num_workers: int,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
        max_img_per_gpu: int = 48,
    ) -> None:
        self.dataset_config = dataset
        self.common_config = common_config
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.max_img_per_gpu = max_img_per_gpu

        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False)

        inner = self._resolve_inner_dataset(self.dataset)
        manifest_samples = getattr(inner, "manifest_samples", None) if inner is not None else None
        if manifest_samples is None:
            raise ValueError(
                "ManifestEvalLoader requires the wrapped RoomEnvelopesDataset to be "
                "constructed with manifest_path=... so per-sample num_views is known."
            )
        self.manifest_num_views = [int(s["num_views"]) for s in manifest_samples]

        # Manifest-driven val: the ManifestBatchSampler emits canonical indices
        # in [0, len(manifest_samples)). The base TupleConcatDataset / common
        # config inherits ``inside_random=True`` from the shared val defaults
        # which, if left on, would discard those indices and replace them with
        # random.randint(0, len_val - 1), overshooting the manifest length and
        # raising IndexError inside ``manifest_samples[seq_index]``. Force it
        # off here and align the inner ``__len__`` with the manifest size so
        # downstream consumers (cumulative_sizes, samplers, progress meters)
        # report the truth.
        n_samples = len(manifest_samples)
        inner.len_train = n_samples
        base_dataset = getattr(self.dataset, "base_dataset", None)
        if base_dataset is not None:
            if getattr(base_dataset, "inside_random", False):
                base_dataset.inside_random = False
            # ConcatDataset.cumulative_sizes was captured at __init__ time with
            # the pre-override len_train. Recompute so len()/bisect_right see
            # the manifest size.
            if hasattr(base_dataset, "datasets") and hasattr(base_dataset, "cumsum"):
                base_dataset.cumulative_sizes = base_dataset.cumsum(base_dataset.datasets)
        if hasattr(self.dataset, "total_samples"):
            self.dataset.total_samples = n_samples
        try:
            common_config.inside_random = False
        except Exception:
            pass

        # Cache the aspect ratio used at val time (val configs use [1.0, 1.0]).
        aspect_range = common_config.augs.aspects
        if aspect_range is None:
            self.aspect_ratio = 1.0
        else:
            self.aspect_ratio = float(aspect_range[0])

        self.batch_sampler = ManifestBatchSampler(
            manifest_num_views=self.manifest_num_views,
            max_img_per_gpu=self.max_img_per_gpu,
            aspect_ratio=self.aspect_ratio,
        )

    @staticmethod
    def _resolve_inner_dataset(wrapped):
        """Return the inner RoomEnvelopesDataset wrapped by ComposedDataset/TupleConcatDataset."""
        ds = getattr(wrapped, "base_dataset", None)
        if ds is None:
            return None
        inner_list = getattr(ds, "datasets", None)
        if not inner_list:
            return None
        return inner_list[0]

    def get_loader(self, epoch):
        # Deterministic, epoch is accepted for API parity with DynamicTorchDataset.
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
        )


class ManifestBatchSampler(Sampler):
    """Yields lists of ``(idx, num_views, aspect_ratio)`` tuples in manifest order.

    Samples are assumed to be pre-sorted by ``num_views`` in the underlying dataset
    so that consecutive same-num_views runs can be packed into batches.
    """

    def __init__(self, manifest_num_views: List[int], max_img_per_gpu: int, aspect_ratio: float = 1.0):
        self.manifest_num_views = list(manifest_num_views)
        self.max_img_per_gpu = int(max_img_per_gpu)
        self.aspect_ratio = float(aspect_ratio)
        self._batches: List[List[Tuple[int, int, float]]] = self._build_batches()

    def _world(self) -> Tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _build_batches(self) -> List[List[Tuple[int, int, float]]]:
        rank, world_size = self._world()
        all_batches: List[List[Tuple[int, int, float]]] = []
        i = 0
        n = len(self.manifest_num_views)
        while i < n:
            nv = self.manifest_num_views[i]
            # Find the end of the contiguous same-nv run.
            j = i
            while j < n and self.manifest_num_views[j] == nv:
                j += 1
            batch_size = max(1, self.max_img_per_gpu // max(1, nv))
            for s in range(i, j, batch_size):
                e = min(s + batch_size, j)
                all_batches.append(
                    [(idx, nv, self.aspect_ratio) for idx in range(s, e)]
                )
            i = j
        # Shard across ranks. Each rank gets every world_size-th batch.
        return all_batches[rank::world_size]

    def __iter__(self):
        for b in self._batches:
            yield b

    def __len__(self) -> int:
        return len(self._batches)

    def set_epoch(self, epoch: int) -> None:  # API parity
        return
