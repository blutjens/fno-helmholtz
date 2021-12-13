from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from pce_pinns.neural_nets import fno_dft as dft

# @torch.jit.script
def expand_core(
    flat_core: torch.Tensor, n_channels: int, nx: int, ny: int
) -> torch.Tensor:
    """Expands core to full Fourier space.

    The Fourier transform, C, of a real-valued tensor in image space obeys the symmetry relation
    ``C[:, i, j, ..., k] = C*[:, (x-i) % x, (y-j) % y, ..., (z-k) % z]``, where ``*`` is the complex
    conjugation and `%` is the modulo operation. Hence there is a more compact representation of
    this tensor, which we call core representation. This function takes a tensor in the core
    representation and expands it out to its Fourier representation.

    More precisely, this function takes ``x * y * n * n`` values and expands them into ``n x n``
    complex matrices at grid positions ``(x, y)`` in the Fourier domain.

    Note that we currently only support odd dimensions due to complications with Nyquist
    frequencies.

    :param flat_core: Input tensor elements of shape ``(x * y * n * n)``.
    :param n_channels: Number of rows / columns, ``n``, of the square matrices.
    :param nx: Size of x dimension.
    :param ny: Size of y dimension.
    :return: Expanded core of shape ``(x, y, n, n, 2)`` where the last dimension is the real and
             imaginary part, respectively.
    """
    if nx % 2 != 1 or ny % 2 != 1:
        raise ValueError("Only odd dimensions are supported.")

    n_params = n_channels * n_channels * nx * ny
    if len(flat_core) != n_params:
        raise ValueError(f"Expected {n_params} but found {len(flat_core)}.")

    core = flat_core.reshape(n_channels, n_channels, -1)

    mid = nx * ny // 2 + 1
    core_real = (
        F.pad(core[..., :mid], [(nx * ny) // 2, 0])
        .reshape(n_channels, n_channels, ny, nx)
        .transpose(2, 3)
    )
    core_imag = (
        F.pad(core[..., mid:], [(nx * ny) // 2 + 1, 0])
        .reshape(n_channels, n_channels, ny, nx)
        .transpose(2, 3)
    )
    assert core_real.shape == core_imag.shape
    assert core_real.shape == (n_channels, n_channels, nx, ny)

    real = torch.fft.ifftshift(core_real + core_real.flip((2, 3)))
    imag = torch.fft.ifftshift(core_imag - core_imag.flip((2, 3)))

    return torch.stack([real, imag], -1).permute(2, 3, 0, 1, 4)


class SpectralConv(nn.Module):
    """Applies convolution to input in truncated Fourier domain.

    Applies convolution to input by transforming into the Fourier domain, applying the convolution
    kernel using matrix multiplication, and inverse transforming. The kernel operates only in the
    last dimension of the ``(+, x, y, n_channels)`` input vector where ``x, y`` refers to the
    spatial dimension that are affected by the 2D Fourier transform. In Fourier domain high
    frequency modes are truncated. Note that the convolution kernel operates in the complex valued
    Fourier domain but is constrained s.t. the result of the inverse Fourier transform is
    real-valued.

    :param n_channels: Number of channels
    :param n_modes: Number of positive frequency modes in Fourier domain, i.e., ``n`` corresponds to
                    a Fourier domain of size ``2n + 1``.
    :param device: The ``torch.device`` on which tensors should be stored.
    """

    def __init__(
        self,
        n_channels: int,
        n_modes: Tuple[int, int],
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        if len(n_modes) != 2:
            raise ValueError("Only 2D Fourier domains are supported.")

        self.device = device
        self.n_channels = n_channels
        self.n_modes = n_modes

        kx, ky = self.n_modes[0], self.n_modes[1]
        n_params = (2 * kx - 1) * (2 * ky - 1) * n_channels * n_channels

        scale = 1.0 / n_channels ** 2
        self.weights = nn.Parameter(scale * torch.rand(n_params, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nx, ny = x.shape[1], x.shape[2]
        kx, ky = self.n_modes[0], self.n_modes[1]

        core = expand_core(
            self.weights,
            self.n_channels,
            2 * kx - 1,
            2 * ky - 1,
        ).unsqueeze(0)

        pf = [
            (
                torch.arange(n, device=self.device) / n,
                torch.cat(
                    [
                        torch.arange(k, device=self.device),
                        torch.arange(n - k + 1, n, device=self.device),
                    ]
                ),
            )
            for n, k in zip((ny, nx), (ky, kx))
        ]
        m = [dft.dft_matrix(p, f) for p, f in pf]
        inv_m = [dft.idft_matrix(p, f) for p, f in pf]

        # transposition is a bit tricky here since rdft2 returns a complex vector:
        #  - transpose(1, -1): (b, x, y, c) -> (b, c, y, x)
        #  - transpose(1, -2): (b, c, fy, fx, 2) -> (b, fx, fy, c, 2)
        fx = dft.rdft2(x.transpose(1, -1), (m[0], m[1])).transpose(1, -2)

        y = dft.cc_matrix_vector_product(core, fx)

        # ... irdft2 takes a complex vector and returns a real-valued vector:
        #  - transpose(1, -2): (b, fx, fy, c, 2) -> (b, c, fy, fx, 2)
        #  - transpose(1, -1): (b, c, fy, fx) -> (b, fx, fy, c)
        return dft.irdft2(y.transpose(1, -2), (inv_m[0], inv_m[1])).transpose(1, -1)


class Layer(nn.Module):
    """FNO Layer.

    Splits input and pushes it separately through the FNO block and an 1D Convolution. Subsequently,
    the results are merged by a non-linear activation function.

    Note that the current implementation takes and returns real-valued tensors and only internally
    deals with complex values.

    :param n_channels: See description in :class:`SpectralConv`.
    :param n_modes: See description in :class:`SpectralConv`.
    :param device: The ``torch.device`` on which tensors should be stored.
    """

    def __init__(
        self,
        n_channels: int,
        n_modes: Tuple[int, int],
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.sconv = SpectralConv(n_channels, n_modes, device=device)
        self.bias = nn.Linear(n_channels, n_channels)#, device=device)
        self.bnorm = nn.BatchNorm2d(n_channels)#, device=device)
        self.activation = nn.ReLU()
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.sconv(x)
        y2 = self.bias(x)
        y = y1 + y2
        z = self.bnorm(y.transpose(1, -1)).transpose(1, -1)
        return self.activation(z)


class FNO(nn.Module):
    """FNO Network.

    TODO write concise summary of FNO

    A real-valued ``(b, x, y, f)`` tensor is transformed into a real-valued ``(b, x, y, c)``
    tensor where ``b`` is the batch size, `x, y` refers to the size of two dimensions subjected to
    the Fourier transform and ``f`` (``c``) is the number of input (output) features (channels).

    Note the difference between (input) features and channels: The former refers to the features,
    arranged in batches and dimensions, which are fed into the network. Internally, these features
    are transformed into channels, where each value of a channel is a weighted superposition of
    input features. Values inside a channel only mix features from the same grid position.

    It is the obligation of the caller to project the channels, that are returned by the model, back
    to feature space.

    :param depth: Number of chained FNO layers.
    :param n_features: Number of input features.
    :param n_channels: See description in :class:`SpectralConv`.
    :param n_modes: See description in :class:`SpectralConv`.
    :param device: The ``torch.device`` on which tensors should be stored.
    """

    def __init__(
        self,
        depth: int,
        n_features: int,
        n_channels: int,
        n_modes: Tuple[int, int],
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        # TODO!!! fix device setting
        fc = nn.Linear(n_features, n_channels)#, device=device)

        layers = [Layer(n_channels, n_modes)]*depth#, device=device)] * depth

        self.f = nn.Sequential(fc, *layers)
        self.n_features = n_features
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"Input tensor has to have {self.n_features} features (size of last dimension), but has {x.shape[-1]} features."
            )

        return self.f(x)
