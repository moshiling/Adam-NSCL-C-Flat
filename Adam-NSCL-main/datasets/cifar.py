"""Split CIFAR-100 dataset for continual-learning experiments.

Splits CIFAR-100 into ``n_tasks`` disjoint tasks, each containing
``100 // n_tasks`` classes.  Labels are remapped to 0-based within
each task so that a shared head with ``n_classes_per_task`` outputs
can be used directly.

Usage::

    task_loaders = build_split_cifar100(
        data_root="/data",
        n_tasks=10,
        batch_size=256,
    )
    for task_id, (train_loader, test_loader) in enumerate(task_loaders):
        ...
"""

from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_split_cifar100(
    data_root: str = "./data",
    n_tasks: int = 10,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: Optional[int] = None,
    download: bool = True,
) -> List[Tuple[DataLoader, DataLoader]]:
    """Build per-task train/test ``DataLoader`` pairs for Split CIFAR-100.

    Args:
        data_root:   Directory where CIFAR-100 will be stored / read from.
        n_tasks:     Number of tasks.  Must divide 100 evenly.  Default: 10.
        batch_size:  Mini-batch size for both train and test loaders.
        num_workers: DataLoader worker processes.
        seed:        Optional random seed for reproducibility.
        download:    Auto-download dataset if not present.

    Returns:
        List of ``(train_loader, test_loader)`` tuples, one per task.
    """
    if 100 % n_tasks != 0:
        raise ValueError(
            f"n_tasks={n_tasks} must divide 100 evenly (CIFAR-100 has 100 classes)."
        )

    classes_per_task = 100 // n_tasks

    if seed is not None:
        torch.manual_seed(seed)

    # ---- Transforms --------------------------------------------------------
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # ---- Datasets ----------------------------------------------------------
    train_ds = CIFAR100(data_root, train=True,  transform=train_transform, download=download)
    test_ds  = CIFAR100(data_root, train=False, transform=test_transform,  download=download)

    train_targets = torch.tensor(train_ds.targets)
    test_targets  = torch.tensor(test_ds.targets)

    # ---- Build per-task class partitions -----------------------------------
    # Classes are assigned in order: task 0 → [0..9], task 1 → [10..19], …
    loaders: List[Tuple[DataLoader, DataLoader]] = []
    for t in range(n_tasks):
        class_start = t * classes_per_task
        task_classes = list(range(class_start, class_start + classes_per_task))

        train_idx = _indices_for_classes(train_targets, task_classes)
        test_idx  = _indices_for_classes(test_targets,  task_classes)

        train_subset = _RemappedSubset(train_ds, train_idx, class_start)
        test_subset  = _RemappedSubset(test_ds,  test_idx,  class_start)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append((train_loader, test_loader))

    return loaders


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _indices_for_classes(
    targets: torch.Tensor, classes: List[int]
) -> List[int]:
    """Return sample indices whose target is in ``classes``."""
    mask = torch.zeros(len(targets), dtype=torch.bool)
    for c in classes:
        mask |= targets == c
    return mask.nonzero(as_tuple=False).squeeze(1).tolist()


class _RemappedSubset(torch.utils.data.Dataset):
    """Dataset subset that remaps absolute class indices to 0-based task labels."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        indices: List[int],
        class_offset: int,
    ) -> None:
        self.dataset = Subset(dataset, indices)
        self.class_offset = class_offset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        return img, label - self.class_offset
