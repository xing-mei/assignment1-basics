import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dataset.ndim == 1 and batch_size > 1
    assert dataset.shape[0] - context_length > 1
    start_indices = np.random.randint(0, dataset.shape[0] - context_length, size=batch_size)
    x = np.zeros((batch_size, context_length), dtype=dataset.dtype)
    y = np.zeros((batch_size, context_length), dtype=dataset.dtype)
    for i, idx in enumerate(start_indices):
        x[i] = dataset[idx:idx + context_length]
        y[i] = dataset[idx + 1 : idx + 1 + context_length]
    
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
