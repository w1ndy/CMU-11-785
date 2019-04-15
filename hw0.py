import math
import torch
import numpy as np

def vectorize_sumproducts(A, B):
    return np.dot(A, B)
def vectorize_Relu(M):
    return np.maximum(0, M)
def vectorize_PrimeRelu(M):
    return np.where(np.array(M) > 0, 1, 0)

def slice_fixed_point(M, start, length):
    return [d[start:start+length] for d in M]
def slice_last_point(M, length):
    return [d[len(d)-length:] for d in M]
def slice_random_point(M, length):
    return [d[np.random.randint(0, len(d)-length):][:length] for d in M]

def pad_pattern_end(M):
    length = max(map(len, M))
    return [(d + (d[::-1] + d) * length)[:length] for d in M]
def result_pad_constant_central(M, c):
    C = [c] * max(map(len, M))
    return [(C + d + C)[math.ceil((len(C)+len(d))/2):][:len(C)] for d in M]

def numpy2tensor(arr):
    return torch.from_numpy(arr)
def tensor2numpy(tensor):
    return tensor.numpy()

def tensor_sumproducts(A, B):
    return torch.dot(A, B)

def tensor_ReLU(M):
    return torch.max(torch.zeros(M.size()), M)
def tensor_ReLU_prime(M):
    return torch.clamp(M, min=0) * torch.reciprocal(torch.clamp(M, min=1e-8))
