import numpy as np
import torch


def find_bit_from_vec(vec):
    vec = vec.reshape(1, -1) if vec.ndim == 1 else vec
    return np.argmax(vec, axis=-1)  # for both continuous & multiple 1s situations


def find_bit_from_mat(mat):
    return np.where(mat)


def top_k_from_vec(vec, k):
    if k == 0:
        return []
    return np.argpartition(vec, -k)[-k:]  # small to large


def top_k_from_mat(mat, k, per_atom=False):
    if k == 0:
        return [[] for _ in range(mat.ndim)]
    flat_indices = np.argpartition(mat.flatten(), -k)[-k:]
    flat_indices = flat_indices[np.argsort(-mat.flatten()[flat_indices], stable=True)]
    indices = np.unravel_index(flat_indices, mat.shape)
    if not per_atom:
        return indices
    else:
        return [np.vstack(indices)[:, i] for i in range(k)]
    

def sigmoid(x, x0=0.0, k=1.0, func=np):
    if func == torch:
        ee = torch.clamp(func.exp(-k * (x - x0)), max=1e5)
    else:
        ee = np.clip(func.exp(-k * (x - x0)), a_min=None, a_max=1e5)
    return 1 / (1 + ee)


def bit_sigmoid(x, func=np):
    k = 2 * np.log(19.0) / 0.6
    return sigmoid(x, x0=0.5, k=k, func=func)  # bit_sigmoid(0.8) == 0.95, bit_sigmoid(0.2) == 0.05


crys_embbeder = {'lat': {}, 'sym': {}, 'coords': {}}
