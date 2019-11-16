import torch
import numpy as np
import math
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

Bbox= [torch.tensor([[[ 825.4611,  423.1288,  884.2031,  445.1735]],[[1176.6071,  452.4636, 1230.8829,  475.8027]]])]
print(Bbox)