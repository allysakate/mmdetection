import torch
import torch.nn as nn
'''
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch
S_min = 10  # Minimum target length, for demonstration purposes
# Initialize random batch of inp vectors, for *size = (T,N,C)
inp = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
print(inp)

# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
print(target)
inp_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
print(inp_lengths)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
print(target_lengths)
ctc_loss = nn.CTCLoss()
loss = ctc_loss(inp, target, inp_lengths, target_lengths)
loss.backward()
'''
x = torch.tensor([[0, 23, 26, 27, 0, 16, 0, 19,  8, 0], #4
        [16, 18, 18, 0, 28,  8, 16, 0, 0, 0], #4
        [ 0,  4,  3,  0, 26, 0, 17, 0,  6, 25], #4
        [13, 18,  6,  8, 29, 0,  5, 0, 0, 28], #3
        [ 0,  3, 23, 24,  7, 0,  9, 0, 19, 27], #3
        [ 7, 19, 26, 18, 19,  6, 21,  0, 0, 16], #2
        [0,  4, 0, 26, 18, 0, 24, 18, 18, 14], #3
        [ 0, 23, 14, 19,  7, 15,  9, 0,  0, 16], #3
        [2, 15,  4, 0,  3, 0,  6,  4,  3, 0]]) #3
print(x)

x_len = torch.tensor([y.nonzero().size(0) for y in x])
print(x_len)
# for l in x_len:
#     print(l.size(0))