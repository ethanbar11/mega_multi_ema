import torch
import time


def with_sparse(X, y):
    X = X.to_sparse()
    start = time.time()
    torch.sparse.mm(X, y)
    end = time.time()
    print('sparse time:', end - start)


def not_sprase(X, y):
    start = time.time()
    X.matmul(y)
    end = time.time()
    print('Not sprase:', end - start)


n = int(1e+4
m = int(1e+4)
print('Creating X')
# Setting torch seed to 42
torch.manual_seed(42)
X = torch.rand(n, m).to('cuda')
mask = X > 0
sparse_X = X * mask
print('Creating y')
y = torch.randn(m, 1).to('cuda')

print('With sparse')
with_sparse(sparse_X, y)
print('Not sparse')
not_sprase(sparse_X, y)
