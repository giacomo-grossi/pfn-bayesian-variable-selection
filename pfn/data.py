"""On-the-fly task generation in parallel CPU workers (no task is seen twice)."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .prior import PriorConfig, sample_batch


def p_max_at(step: int, total: int, curriculum) -> int:
    """Curriculum on the maximal dimension: equal thirds of training (Table 14)."""
    k = len(curriculum)
    return int(curriculum[min(k - 1, step * k // max(total, 1))])


class TaskStream(IterableDataset):
    """Yields batches (dicts of tensors). All tasks in a batch share (N, n, p)."""

    def __init__(self, prior: PriorConfig, total_steps: int, curriculum=(50, 200, 500),
                 max_cells=3.5e5, max_batch=64, seed=0, step_counter=None):
        super().__init__()
        self.prior = prior
        self.total = total_steps
        self.curriculum = curriculum
        self.max_cells = max_cells
        self.max_batch = max_batch
        self.seed = seed
        self.step_counter = step_counter if step_counter is not None else mp.Value("i", 0)

    def _batch_meta(self, rng):
        pmax = p_max_at(self.step_counter.value, self.total, self.curriculum)
        cfg = self.prior.replace(p_max=pmax)
        return sample_batch(rng, cfg, max_cells=self.max_cells, max_batch=self.max_batch)

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = np.random.default_rng([self.seed, wid, np.random.SeedSequence().entropy % (2**31)])
        while True:
            b = self._batch_meta(rng)
            yield {"X": torch.from_numpy(b["X"]), "y": torch.from_numpy(b["y"]),
                   "gamma": torch.from_numpy(b["gamma"]), "n": int(b["n"])}


def _single_thread_worker(_):
    """Each data worker uses one BLAS/torch thread: with k workers on k cores, multithreaded
    linear algebra inside every worker would oversubscribe the CPU."""
    torch.set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def make_loader(stream: TaskStream, workers: int):
    return torch.utils.data.DataLoader(stream, batch_size=None, num_workers=workers,
                                       pin_memory=torch.cuda.is_available(),
                                       persistent_workers=workers > 0,
                                       prefetch_factor=4 if workers > 0 else None,
                                       worker_init_fn=_single_thread_worker if workers > 0 else None)


def estimate_borders(prior: PriorConfig, n_bins=256, n_batches=300, seed=123, curriculum=(50,)):
    """Bin edges of the bar distribution from quantiles of standardized responses of the prior."""
    rng = np.random.default_rng(seed)
    ys = []
    for _ in range(n_batches):
        b = sample_batch(rng, prior.replace(p_max=max(curriculum)), max_cells=2e4, max_batch=4)
        y = b["y"]
        ys.append(y[:, b["n"]:].ravel())
    y = np.concatenate(ys)
    # quantile edges; the outer bins [min, q1] and [q_{M-1}, max] carry the half-normal tails
    qs = np.quantile(y, np.linspace(0.0, 1.0, n_bins + 1))
    qs = np.maximum.accumulate(qs + np.arange(qs.size) * 1e-6)   # strictly increasing
    return torch.tensor(qs, dtype=torch.float32)
