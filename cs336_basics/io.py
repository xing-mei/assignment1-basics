import os
import typing
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

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    ckpt_obj = {
        'model_states': model.state_dict(),
        'optimizer_states': optimizer.state_dict(),
        'current_iteration': iteration,
    }
    torch.save(ckpt_obj, out)

def load_checkpoint(
    ckpt_file: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],        
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    assert model is not None
    assert optimizer is not None
    ckpt_obj = torch.load(ckpt_file)
    model.load_state_dict(ckpt_obj['model_states'])
    optimizer.load_state_dict(ckpt_obj['optimizer_states'])
    return ckpt_obj['current_iteration']
