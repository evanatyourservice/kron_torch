# PSGD Kron

For original PSGD repo, see [psgd_torch](https://github.com/lixilinx/psgd_torch).

For JAX version, see [psgd_jax](https://github.com/evanatyourservice/psgd_jax).

Implementation of [PSGD Kron optimizer](https://github.com/lixilinx/psgd_torch) in PyTorch. 
PSGD is a second-order optimizer originally created by Xi-Lin Li that uses either a hessian-based 
or whitening-based (gg^T) preconditioner and lie groups to improve training convergence, 
generalization, and efficiency. I highly suggest taking a look at Xi-Lin's PSGD repo's readme linked
to above for interesting details on how PSGD works and experiments using PSGD.

### `kron`:

The most versatile and easy-to-use PSGD optimizer is `kron`, which uses a 
Kronecker-factored preconditioner. It has less hyperparameters that need tuning than adam, and can 
be a drop-in replacement for adam. It keeps a dim's preconditioner as either triangular 
or diagonal based on `max_size_triangular` and `max_skew_triangular`. For example, for a layer 
with shape (256, 128, 64), triangular preconditioners would be shapes (256, 256), (128, 128), and 
(64, 64) and diagonal preconditioners would be shapes (256,), (128,), and (64,). Depending on how 
these two settings are chosen, `kron` can balance between memory/speed and performance (see below).


## Installation

```bash
pip install kron-torch
```

## Basic Usage (Kron)

Kron schedules the preconditioner update probability by default to start at 1.0 and anneal to 0.03 
at the beginning of training, so training will be slightly slower at the start but will speed up 
to near adam's speed by around 3k steps.

For basic usage, use `kron` optimizer like any other pytorch optimizer:

```python
from kron_torch import Kron

optimizer = Kron(params)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Basic hyperparameters:**

TLDR: Learning rate and weight decay act similarly to adam, but lr might be able to be a little 
higher like 0.001 -> 0.002 or 0.003, and weight decay might be best a little lower like 
0.1 -> 0.05 or 0.03. There is no b2 or epsilon.

`learning_rate`: Kron's learning rate acts similarly to adam's, but can withstand a higher 
learning rate. Try setting 3x higher. If 0.001 was best for adam, try setting kron's to 0.003.

`weight_decay`: PSGD does not rely on weight decay for generalization as much as adam, and too
high weight decay can hurt performance. Try setting 3-10x lower. If the best weight decay for 
adam was 0.1, you can set kron's to 0.03 or 0.01.

`max_size_triangular`: Anything above this value will have a diagonal preconditioner, anything 
below will have a triangular preconditioner. So if you have a dim with size 16,384 that you want 
to use a diagonal preconditioner for, set `max_size_triangular` to something like 15,000. Default 
is 8192.

`max_skew_triangular`: Any tensor with skew above this value with make the larger dim diagonal.
For example, with the default value for `max_skew_triangular` as 10, a bias layer of shape 
(256,) would be diagonal because 256/1 > 10, and an embedding dim of shape (50000, 768) would 
be (diag, tri) because 50000/768 is greater than 10. The default value of 10 usually makes 
layers like bias, scale, and vocab embedding use diagonal with the rest as triangular.

`min_ndim_triangular`: Any tensor with less than this number of dims will have all diagonal 
preconditioners. Default is 2, so single-dim tensors like bias and scale will use diagonal.

Interesting note: Setting `max_skew_triangular` to 1 will make the largest dim have a diagonal 
preconditioner and the rest have triangular, which usually uses slightly less memory than adam. 
Setting `max_size_triangular` to 0 will make all layers have diagonal preconditioners which uses 
the least memory and runs the fastest, but performance might be worse.

`preconditioner_update_probability`: Preconditioner update probability uses a schedule by default 
that works well for most cases. It anneals from 1 to 0.03 at the beginning of training, so training 
will be slightly slower at the start but will speed up to near adam's speed by around 3k steps.

See kron.py for more hyperparameter details.