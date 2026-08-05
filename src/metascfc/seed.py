import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(seed: Optional[int] = None) -> callable:
    if seed is not None:

        def _init(worker_id: int) -> None:
            np.random.seed(seed + worker_id)

        return _init
    return None
