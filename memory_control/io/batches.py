from __future__ import annotations

from pathlib import Path
from typing import Any

from model_zoo import make_synthetic_batch

from memory_control.core.config import RunConfig


class BatchProvider:
    def next(self, batch_size: int) -> tuple[Any, Any]:
        raise NotImplementedError


class SyntheticBatchProvider(BatchProvider):
    def __init__(self, profile: Any, device: Any, torch: Any) -> None:
        self.profile = profile
        self.device = device
        self.torch = torch

    def next(self, batch_size: int) -> tuple[Any, Any]:
        return make_synthetic_batch(self.profile, batch_size, self.device, self.torch)


class DataLoaderBatchProvider(BatchProvider):
    def __init__(self, loader: Any, device: Any, sampler: Any | None = None) -> None:
        self.loader = loader
        self.device = device
        self.sampler = sampler
        self.epoch = 0
        self.iterator = self._new_iterator()

    def _new_iterator(self) -> Any:
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self.epoch)
            self.epoch += 1
        return iter(self.loader)

    def next(self, batch_size: int) -> tuple[Any, Any]:
        try:
            x, y = next(self.iterator)
        except StopIteration:
            self.iterator = self._new_iterator()
            x, y = next(self.iterator)
        x = x[:batch_size].to(self.device, non_blocking=True)
        y = y[:batch_size].to(self.device, non_blocking=True)
        return x, y


def build_batch_provider(
    cfg: RunConfig,
    profile: Any,
    device: Any,
    torch: Any,
    *,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
) -> BatchProvider:
    if cfg.dataset_name == "synthetic":
        return SyntheticBatchProvider(profile, device, torch)

    if cfg.dataset_name != "cifar10":
        raise SystemExit(f"unknown dataset_name={cfg.dataset_name!r}")

    if tuple(profile.input_shape) != (3, 32, 32):
        raise SystemExit(
            "dataset_name=cifar10 requires a model profile with input_shape=(3, 32, 32). "
            "Use model_name=tiny_cnn or add a compatible model_zoo entry."
        )

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit(
            "dataset_name=cifar10 requires torchvision. Install cloud dependencies with: "
            "python -m pip install -r requirements-cloud.txt"
        ) from exc

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    dataset = datasets.CIFAR10(
        root=str(Path(cfg.data_dir).expanduser()),
        train=True,
        download=bool(cfg.download_dataset),
        transform=transform,
    )
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    loader_batch_size = max(1, int(cfg.max_micro_batch))
    sampler = None
    shuffle = True
    if distributed_world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=int(distributed_world_size),
            rank=int(distributed_rank),
            shuffle=True,
            seed=int(cfg.seed),
            drop_last=True,
        )
        shuffle = False
    return DataLoaderBatchProvider(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=loader_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            num_workers=max(0, int(cfg.num_workers)),
            pin_memory=bool(cfg.pin_memory and device.type == "cuda"),
            persistent_workers=bool(int(cfg.num_workers) > 0),
            generator=None if sampler is not None else generator,
        ),
        device,
        sampler=sampler,
    )
