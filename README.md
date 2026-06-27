# KNN_CUDA (reborn)

> **Community re-upload of [`unlimblue/KNN_CUDA`](https://github.com/unlimblue/KNN_CUDA).**
> The original `unlimblue` account and its release wheel
> (`KNN_CUDA-0.2-py3-none-any.whl`) are no longer available on GitHub. This
> repository reconstructs the package **verbatim** from an existing local
> installation so the 3D computer-vision research community can keep using it.
>
> **I claim no ownership of this code — all rights belong to the original
> authors.** See [DISCLAIMER.md](DISCLAIMER.md) for full attribution and the
> source-file headers for the original copyright.

GPU-accelerated **k-nearest-neighbors** for PyTorch tensors, implemented as a
CUDA extension.

---

## Why this fork exists

Several popular point-cloud autoencoder repos — notably
[**Point-MAE**](https://github.com/Pang-Yatian/Point-MAE) and
[**Point-M2AE**](https://github.com/ZrrSkywalker/Point-M2AE) — list this in
their install instructions:

```bash
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

That URL now 404s because the `unlimblue` account is gone. This repo provides a
drop-in replacement wheel so those install steps work again.

## How it works

The published wheel is **pure Python** (`py3-none-any`): it ships the C++/CUDA
source files inside the package and compiles them **on first import** using
`torch.utils.cpp_extension.load` (Just-In-Time compilation). This is why a
single wheel works across CUDA / PyTorch versions — the kernels are built
against *your* environment the first time you `import knn_cuda`.

Consequences:

- A **CUDA toolkit** (`nvcc`) and a C++ compiler must be available at runtime.
- [`ninja`](https://pypi.org/project/ninja/) is required for the JIT build
  (installed automatically as a dependency).
- The **first** `import knn_cuda` is slow (it compiles); later imports are fast
  because the build is cached.

## Requirements

- Linux
- Python ≥ 3.6
- PyTorch with CUDA support (matching your installed CUDA toolkit)
- NVIDIA CUDA toolkit (`nvcc` on `PATH`, e.g. `/usr/local/cuda*/bin`)
- `ninja`

## Installation

### From a release wheel (recommended — the drop-in replacement)

```bash
pip install --upgrade https://github.com/altaykacan/KNN_CUDA_reborn/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### From source

```bash
git clone https://github.com/altaykacan/KNN_CUDA_reborn.git
cd KNN_CUDA_reborn
pip install .
```

### Build the wheel yourself

```bash
pip install build
python -m build --wheel
# -> dist/KNN_CUDA-0.2-py3-none-any.whl
```

Because the wheel only carries source (not compiled binaries), the wheel you
build is platform-independent and identical in spirit to the original release.

## Usage

```python
import torch
from knn_cuda import KNN

# k neighbors; transpose_mode=True => inputs/outputs are (B, N, C)-style
knn = KNN(k=10, transpose_mode=True)

ref   = torch.rand(32, 1000, 5).cuda()   # (batch, n_ref,   channels)
query = torch.rand(32,  50,  5).cuda()   # (batch, n_query, channels)

dist, indx = knn(ref, query)   # dist, indx: (32, 50, 10)
```

### API

```python
KNN(k, transpose_mode=False)
```

- **`k`** — number of nearest neighbors to return.
- **`transpose_mode`**
  - `False` (default): tensors are `(B, C, N)` — channels first. Outputs are
    `(B, k, n_query)`.
  - `True`: tensors are `(B, N, C)` — points first (the common point-cloud
    layout). Outputs are `(B, n_query, k)`.

`forward(ref, query)` returns `(dist, indx)` where `dist` are the Euclidean
distances to the `k` nearest reference points for each query point and `indx`
are the corresponding **0-based** indices into `ref`.

> **Note:** the reference / query batch dimensions must match
> (`ref.size(0) == query.size(0)`).

## Attribution

- Re-upload of `unlimblue/KNN_CUDA` (original account/wheel no longer on GitHub).
- CUDA kernels are a modified version of
  [`vincentfpgarcia/kNN-CUDA`](https://github.com/vincentfpgarcia/kNN-CUDA),
  with modifications attributed to Christopher B. Choy
  (`chrischoy@ai.stanford.edu`); see the header in
  [`knn_cuda/csrc/cuda/knn.cu`](knn_cuda/csrc/cuda/knn.cu).

## License

The [`LICENSE`](LICENSE) (MIT) covers **only the packaging contributions** in
this repository. The original kNN source code remains under the license and
ownership of its respective authors. See [DISCLAIMER.md](DISCLAIMER.md).
