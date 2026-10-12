import numpy as np
import torch


def worker_init_fn(worker_id):
    """
    Give each DataLoader worker an independent,
    reproducible NumPy random-number generator.
    """
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:
        return

    base_seed = torch.initial_seed() % (2**32)

    worker_info.dataset.set_rng(
        np.random.default_rng(base_seed + worker_id)
    )