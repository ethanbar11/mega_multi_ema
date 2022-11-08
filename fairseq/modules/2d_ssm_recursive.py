# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fairseq.incremental_decoding_utils import with_incremental_state


@with_incremental_state
class TwoDimensionalSSM(nn.Module):
    """Exponential Moving Average Layer.

    See "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(
            self,
            embed_dim,
            ndim=2,
            bidirectional=False,
            truncation=None,
    ):
        super().__init__()
        self.is_2_dim = True
        self.truncation = truncation
        self.embed_dim = embed_dim
        self.ndim = ndim

        # TODO: Add support in ndim>1 bidirectionality, and truncation
        # self.bidirectional = bidirectional
        self.scale = math.sqrt(1.0 / self.ndim)

        # D x N x 1
        self.A1 = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim))
        self.A2 = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim))
        self.A3 = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim))
        self.A4 = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim))
        self.B1 = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim))
        self.B2 = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim))

        # D x N
        self.C1 = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim))
        self.C2 = nn.Parameter(torch.Tensor(self.embed_dim, self.ndim))

        # sized D because this is a residual connection (element-wise)
        self.omega = nn.Parameter(torch.Tensor(embed_dim))

        self._kernel = None
        self._coeffs = None

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha
            nn.init.normal_(self.A1, mean=0.0, std=0.2)
            nn.init.normal_(self.A2, mean=0.0, std=0.2)
            nn.init.normal_(self.A3, mean=0.0, std=0.2)
            nn.init.normal_(self.A4, mean=0.0, std=0.2)
            nn.init.normal_(self.B1, mean=0.0, std=0.2)
            nn.init.normal_(self.B2, mean=0.0, std=0.2)
            # TODO: After expanding to n_dim>1 , checkout what's being done with beta in EMA

            nn.init.normal_(self.C1, mean=0.0, std=1.0)
            nn.init.normal_(self.C2, mean=0.0, std=1.0)

            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        A1 = torch.sigmoid(self.A1) / 2
        A2 = torch.sigmoid(self.A2) / 2
        A3 = torch.sigmoid(self.A3) / 2
        A4 = torch.sigmoid(self.A4) / 2
        B1 = torch.sigmoid(self.B1) / 2
        B2 = torch.sigmoid(self.B2) / 2
        return A1, A2, A3, A4, B1, B2

    def compute_x_matrix(self, length):
        # D each
        A1, A2, A3, A4, B1, B2 = self._calc_coeffs()

        # l x l x L x D x n
        x_h = torch.zeros(length, length, length ** 2, self.embed_dim, self.ndim)
        x_v = torch.zeros(length, length, length ** 2, self.embed_dim, self.ndim)
        zeros_vec = torch.zeros(length ** 2, self.embed_dim, self.ndim)
        for i in range(length):
            for j in range(length):
                # L x D x n
                x_h_i_minus_j = x_h[i, j - 1] if j - 1 >= 0 else zeros_vec
                x_v_i_minus_j = x_v[i, j - 1] if j - 1 >= 0 else zeros_vec

                x_h[i, j] = A1 * x_h_i_minus_j + A2 * x_v_i_minus_j
                x_h[i, j, i * length + j] = B1

                x_h_minus_i_j = x_h[i - 1, j] if i - 1 >= 0 else zeros_vec
                x_v_minus_i_j = x_v[i - 1, j] if i - 1 >= 0 else zeros_vec

                x_v[i, j] = A3 * x_h_minus_i_j + A4 * x_v_minus_i_j
                x_v[i, j, i * length + j] = B2
        return x_h, x_v

    def _compute_kernel(self, length: int):
        self._kernel = None

        # l x l x L x D x N
        x_h_matrix, x_v_matrix = self.compute_x_matrix(length)
        # L x L x D x N
        x_h_matrix = x_h_matrix.reshape(length ** 2, length ** 2, self.embed_dim, self.ndim)
        x_v_matrix = x_v_matrix.reshape(length ** 2, length ** 2, self.embed_dim, self.ndim)

        # L x L x H
        output_horizontal = torch.einsum("l k D N ,H N ->l k H", x_h_matrix, self.C1)
        output_vertical = torch.einsum("l k D N ,H N ->l k H", x_v_matrix, self.C2)

        # L x L x H
        output = output_horizontal + output_vertical

        return output

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def kernel(self, length: int):
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        if self.training:
            return self._compute_kernel(kernel_size)
        else:
            if self._kernel is None or self._kernel.size(-1) < kernel_size:
                self._kernel = self._compute_kernel(kernel_size)
            return self._kernel[..., :kernel_size]

    def step(self, x, length, hx=None):
        if length == 1:
            return self.one_step(x, hx=hx)

        # D x N x 1
        p, q = self.coeffs()
        # D x N x L+1
        vander = torch.arange(length + 1).to(p).view(1, 1, length + 1) * torch.log(q)
        vander = torch.exp(vander)
        if hx is not None:
            # D x N x L * D x N x 1 -> D x N x L
            k = vander[:, :, 1:] * (self.gamma * self.scale).unsqueeze(-1)
            ox = torch.einsum('bdn,dnl->bdl', hx, k)
            # D x N * B x D x N -> B x D x N
            hh = vander[:, :, -1] * hx
        else:
            ox = None
            hh = None

        # D x N x L
        vander = vander[:, :, :-1]
        kernel = (p * self.beta) * vander
        k = torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

        k_f = torch.fft.rfft(k.float(), n=2 * length)
        x_f = torch.fft.rfft(x.float(), n=2 * length)
        # B x D x L
        out = torch.fft.irfft(x_f * k_f, n=2 * length)[..., 0:length]
        out = out.type_as(x)
        if ox is not None:
            out = out + ox

        h = torch.einsum('bdl,dnl->bdn', x, torch.flip(kernel, dims=[2]))
        if hh is not None:
            h = h + hh
        # L x B x D, B x D x N
        return out.permute(2, 0, 1), h

    def one_step(self, x, hx=None):
        p, q = self.coeffs()
        # (D x N) x (B x D x 1) -> B x D x N
        h = (p * self.beta).squeeze(-1) * x
        if hx is not None:
            h = h + q.squeeze(-1) * hx
        # B x D
        out = torch.einsum('bdn,dn->bd', h, self.gamma * self.scale)
        # 1 x B x D, B x D x N
        return out.unsqueeze(0), h

    def forward(
            self,
            x,
            padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """

        L, bsz, embed_dim = x.size()
        l = int(math.sqrt(L))
        assert embed_dim == self.embed_dim

        # L x B x D
        residual = x * self.omega

        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        # L x L x D
        k = self.kernel(l)

        # if self.bidirectional:
        #     k1, k2 = torch.split(k, [self.embed_dim, self.embed_dim], dim=0)
        #     # D x 2*L-1
        #     k = F.pad(k1, (kernel_size - 1, 0)) + F.pad(k2.flip(-1), (0, kernel_size - 1))
        #     x = F.pad(x, (kernel_size - 1, 0))
        #     fft_len = fft_len + kernel_size - 1
        #     s = 2 * kernel_size - 2

        # k_f = torch.fft.rfft(k.float(), n=2 * fft_len)
        # x_f = torch.fft.rfft(x.float(), n=2 * fft_len)
        # B x D x L
        out = torch.fft.irfft(x_f * k_f, n=2 * fft_len)[..., s:s + seq_len]
        out = out.type_as(x)
        # B x D x L -> L x B x D
        out = F.silu(out.permute(2, 0, 1) + residual)

        return out

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]) -> Dict[
        str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "ema_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
                          buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, "ema_state", buffer)

    @torch.jit.export
    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def extra_repr(self) -> str:
        return 'edim={}, ndim={}, bidirectional={}, trunction={}'.format(self.embed_dim, self.ndim, self.bidirectional,
                                                                         self.truncation)


def test_step_and_matrix():
    embed_dim = 5
    ssm = TwoDimensionalSSM(embed_dim)
    L = 4
    B = 1
    # L x B x H
    x = torch.randn(L, B, embed_dim)
    y = ssm(x)


if __name__ == '__main__':
    test_step_and_matrix()
