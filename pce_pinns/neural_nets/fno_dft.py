import math
from typing import Tuple

import torch

def cc_matrix_vector_product(m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Returns the product of a complex matrix and a complex vector.

    :param m: Matrix as ``(*, m, n, 2)`` tensor.
    :param x: Vector as ``(*, n, 2)`` tensor.
    :return: Result as ``(*, m, 2)`` tensor.
    """
    xr = x[..., 0:1]
    xi = x[..., 1:2]
    mr = m[..., 0]
    mi = m[..., 1]

    rr = torch.matmul(mr, xr)
    ii = torch.matmul(mi, xi)
    ri = torch.matmul(mr, xi)
    ir = torch.matmul(mi, xr)

    return torch.cat([rr - ii, ri + ir], dim=-1)


def cr_matrix_vector_product(m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Returns the product of a complex matrix and a real vector.

    :param m: Matrix as ``(*, m, n, 2)`` tensor.
    :param x: Vector as ``(*, n)`` tensor.
    :return: Result as ``(*, m, 2)`` tensor.
    """
    xr = x.unsqueeze(-1)
    mr = m[..., 0]
    mi = m[..., 1]

    rr = torch.matmul(mr, xr)
    ir = torch.matmul(mi, xr)

    return torch.cat([rr, ir], dim=-1)


def dft_matrix(p: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Returns matrix that maps vector from time domain into frequency domain via DFT.

    Note that the dimensions of ``p`` and ``f`` do not have to match.

    :param p: Vector with ``n`` points in time domain.
    :param f: Vector with ``m`` points in frequency domain.
    :return: ``m x n`` DFT matrix.
    """
    pf = torch.outer(f, p)
    arg = -pf * 2.0 * math.pi
    n = p.shape[-1]
    return torch.stack([torch.cos(arg), torch.sin(arg)], dim=-1) / n


def idft_matrix(p: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Returns matrix that maps vector from frequency domain into time domain via (inverse) DFT.

    Note that the dimensions of ``p`` and ``f`` do not have to match.

    :param p: Vector with ``n`` points in time domain.
    :param f: Vector with ``m`` points in frequency domain.
    :return: (inverse) ``n x m`` DFT matrix.
    """
    fp = torch.outer(p, f)
    arg = fp * 2.0 * math.pi
    return torch.stack([torch.cos(arg), torch.sin(arg)], dim=-1)


def rdft2(x: torch.Tensor, m: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """DFT transform for real-valued 2D input vectors.

    Applies a DFT transform to all dimensions of batched, real-valued input if ``m`` is a proper DFT
    matrix.

    :param x: Input vector as ``(*, x, y)`` tensor.
    :param m: DFT matrix for dimension ``x, y``.
    :return: Complex DFT transform of input as ``(*, fx, fy, 2)`` tensor where the last dimension
             wraps the real and imaginary part, respectively.
    """
    x = cr_matrix_vector_product(m[0], x.transpose(-2, -1)).transpose(-3, -2)
    x = cc_matrix_vector_product(m[1], x)
    return x


def irdft2(x: torch.Tensor, m: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Inverse DFT transform for real-valued 2D input vectors.

    Applies an inverse DFT transform to all dimensions of batched, real-valued input if ``m`` is a
    proper inverse DFT matrix.

    :param x: Input vector as ``(*, x, y)`` tensor.
    :param m: Inverse DFT matrix for dimension ``x, y``.
    :return: Complex DFT transform of input as ``(*, fx, fy)`` tensor.
    """
    x = cc_matrix_vector_product(m[0], x.transpose(-3, -2)).transpose(-3, -2)
    x = cc_matrix_vector_product(m[1], x)
    return x[..., 0]
