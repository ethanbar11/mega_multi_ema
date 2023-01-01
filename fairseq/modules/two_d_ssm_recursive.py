# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple
from einops import rearrange, einsum
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn

from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.ssm_coefficient import CoeffCalculator


def plot_heatmap(x, title):
    import matplotlib.pyplot as plt
    plt.imshow(x.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.show()


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
            L=32 ** 2,
            force_coeff_calc=False,
    ):
        super().__init__()
        print(L)
        self.is_2_dim = True
        self.truncation = truncation
        self.embed_dim = embed_dim
        self.bidirectional = False
        self.ndim = ndim

        # TODO: Add support in ndim>1 bidirectionality, and truncation
        self.bidirectional = bidirectional
        self.scale = math.sqrt(1.0 / self.ndim)
        self.kernel_dim = 4 * embed_dim if self.bidirectional else embed_dim

        # TODO: Change this where we'll work with other benchmarks
        self.one_side_length = int(math.sqrt(L))
        self.coeff_calc = CoeffCalculator(self.one_side_length)
        self.coeff_calc.calc_coeffs_lazy(force=force_coeff_calc)
        self.matrices = self.coeff_calc.matrices
        for key, inner_dic in self.matrices.items():
            for symbol, matrix in inner_dic.items():
                self.matrices[key][symbol] = matrix.cuda()
        # self.matrices = nn.ParameterDict(self.matrices)

        # D x N x 1
        self.A = {
            'A_1': nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim)),
            'A_2': nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim)),
            'A_3': nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim)),
            'A_4': nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim)),
        }
        self.A = nn.ParameterDict(self.A)
        self.B_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))
        self.B_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))
        # D x N
        self.C_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))
        self.C_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))

        # sized D because this is a residual connection (element-wise)
        self.omega = nn.Parameter(torch.Tensor(embed_dim))

        self.horizontal_flow = None
        self.vertical_flow = None

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
            for symbol, tensor in self.A.items():
                nn.init.normal_(tensor, mean=0, std=0.2)
            nn.init.normal_(self.B_1, mean=0.0, std=0.2)
            nn.init.normal_(self.B_2, mean=0.0, std=0.2)
            # TODO: After expanding to n_dim>1 , checkout what's being done with beta in EMA

            nn.init.normal_(self.C_1, mean=0.0, std=1.0)
            nn.init.normal_(self.C_2, mean=0.0, std=1.0)

            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        A = {}
        for symbol, tensor in self.A.items():
            A[symbol] = torch.sigmoid(tensor)
        B1 = torch.sigmoid(self.B_1)
        B2 = torch.sigmoid(self.B_2)
        return A, B1, B2

    def compute_x_matrix(self, kernel_dim):
        # H x N each
        A, B1, B2 = self._calc_coeffs()
        power_dim = kernel_dim * 2
        # l x l  D x N
        A_powers = {}
        for symbol, tensor in A.items():
            A_powers[symbol] = torch.exp(
                einsum(torch.arange(power_dim).to(tensor.device),
                       torch.log(tensor),
                       'l , h n -> l h n'))
        B = torch.stack([B1, B2], dim=0)
        outputs = {}
        for direction in self.matrices.keys():
            # Should be sized R x H x N
            outputs[direction] = None
        for direction, matrices in self.matrices.items():
            output = outputs[direction]
            for symbol, matrix in matrices.items():
                vec = B if symbol == 'B' else A_powers[symbol]
                current_calculation = einsum(matrix, vec, 'R V, V h n -> R h n')
                # Replaces all the zeros in  current_calculation
                # with ones so the multiplication will be valid
                if output is None:
                    output = current_calculation
                else:
                    current_calculation[current_calculation == 0] = 1
                    # TODO:  Maybe inserting bug here with the 1 replacement.
                    output[output == 0] = 1
                    output = output * current_calculation
            output[output == 1] =0
            outputs[direction] = output
        for direction, matrix in outputs.items():
            outputs[direction] = rearrange(matrix, '(r1 r2) h n -> r1 r2 h n',
                                           r1=self.one_side_length ** 2,
                                           r2=self.coeff_calc.coeff_rows_amount // (self.one_side_length ** 2))
            # Sum over the second dimension
            outputs[direction] = torch.sum(outputs[direction], dim=1)
        return outputs

    def _compute_kernel(self, length: int):
        self._kernel = None
        A, B_1, B_2 = self._calc_coeffs()
        # l x l x D x N
        outputs = self.compute_x_matrix(length)
        # L x L x D x N

        # L x L x H
        output_horizontal = einsum(outputs['horizontal'], self.C_1 * self.scale, "l H N ,H N ->l H")
        output_vertical = einsum(outputs['vertical'], self.C_2 * self.scale, "l H N ,H N ->l H")
        # L x L x H
        output = output_horizontal + output_vertical
        output = output.view(length, length, self.kernel_dim)
        output[0, :, :, ] *= 2
        output[:, 0, :, ] *= 2
        output[0, 0] /= 4

        return output

    def compute_sympy_kernel(self):
        A, B1, B2 = self._calc_coeffs()
        return self.coeff_calc.compute_sympy_kernel(A, B1, B2, self.C_1, self.C_2)

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
            return self._kernel

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

    def one_step(self, x, state_h_left, state_v_left=None, state_h_up=None, state_v_up=None,
                 bidirectional_index=None):

        scalar_left = 1 if state_h_left is None or state_v_left is None else 1 / 2
        scalar_up = 1 if state_h_up is None or state_v_up is None else 1 / 2
        state_h_left = state_h_left if state_h_left is not None else torch.zeros(self.embed_dim, self.ndim).to(x)
        state_v_left = state_v_left if state_v_left is not None else torch.zeros(self.embed_dim, self.ndim).to(x)
        state_h_up = state_h_up if state_h_up is not None else torch.zeros(self.embed_dim, self.ndim).to(x)
        state_v_up = state_v_up if state_v_up is not None else torch.zeros(self.embed_dim, self.ndim).to(x)
        # X is sized B x D
        B = x.shape[0]
        D = x.shape[1]

        # D x N x 1
        A1, A2, B1, B2 = self.coeffs()
        if bidirectional_index is not None:
            A1 = A1[self.embed_dim * bidirectional_index:self.embed_dim * (bidirectional_index + 1), :]
            A2 = A2[self.embed_dim * bidirectional_index:self.embed_dim * (bidirectional_index + 1), :]
            B1 = B1[self.embed_dim * bidirectional_index:self.embed_dim * (bidirectional_index + 1), :]
            B2 = B2[self.embed_dim * bidirectional_index:self.embed_dim * (bidirectional_index + 1), :]
            C1 = self.C_1[self.embed_dim * bidirectional_index:self.embed_dim * (bidirectional_index + 1), :]
            C2 = self.C_2[self.embed_dim * bidirectional_index:self.embed_dim * (bidirectional_index + 1), :]
        else:
            C1 = self.C_1
            C2 = self.C_2

        next_state_h = (A1 * state_h_left + A2 * state_v_left) * scalar_left + B1 * x
        next_state_v = (A2 * state_h_up + A1 * state_v_up) * scalar_up + B2 * x

        out1 = torch.einsum('bdn,dn -> bd', next_state_h, C1 * self.scale)
        out2 = torch.einsum('bdn,dn -> bd', next_state_v, C2 * self.scale)
        # B x D
        out = out1 + out2
        return out, next_state_h, next_state_v

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

        seq_len, bsz, embed_dim = x.size()

        assert embed_dim == self.embed_dim

        # L x B x D
        residual = x * self.omega

        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        assert not self.bidirectional or incremental_state is None, 'Bidirectional EMA does not support incremental state'
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_state' in saved_state:
                h = saved_state['prev_state']
            else:
                h = None
            out, h = self.step(x, seq_len, hx=h)
            saved_state['prev_state'] = h
            self._set_input_buffer(incremental_state, saved_state)
            # B x D -> 1 x B x D
            out = F.silu(out + residual)
        else:
            # D x L
            fft_len = seq_len
            fft_len = int(math.sqrt(fft_len))
            k = self.kernel(fft_len).permute(2, 0, 1)  # H x L x L
            s = 0
            kernel_size = k.size(1)
            x = x.view(bsz, embed_dim, int(math.sqrt(seq_len)), int(math.sqrt(seq_len)))
            two_dim_seq_len = int(math.sqrt(seq_len))
            out = None
            if self.bidirectional:
                kernels = list(
                    torch.split(k, [self.embed_dim for i in range(4)], dim=0))  # 4 kernels, one for each direction.
                flip_dims = [[], [-2], [-1], [-2, -1]]
                for idx, flip in enumerate(flip_dims):
                    k = kernels[idx]
                    curr_x = torch.flip(x, dims=flip)

                    k_f = torch.fft.rfft2(k.float(), s=(2 * fft_len, 2 * fft_len))
                    x_f = torch.fft.rfft2(curr_x.float(), s=(2 * fft_len, 2 * fft_len))
                    curr = torch.fft.irfft2(x_f * k_f, s=(2 * fft_len, 2 * fft_len))[..., s:two_dim_seq_len + s,
                           s:two_dim_seq_len + s]
                    curr_after_flip = torch.flip(curr, dims=flip)
                    if out is None:
                        out = curr_after_flip
                    else:
                        out += curr_after_flip
            else:
                k_f = torch.fft.rfft2(k.float(), s=(2 * fft_len, 2 * fft_len))
                x_f = torch.fft.rfft2(x.float(), s=(2 * fft_len, 2 * fft_len))
                out = torch.fft.irfft2(x_f * k_f, s=(2 * fft_len, 2 * fft_len))[..., s:two_dim_seq_len + s,
                      s:two_dim_seq_len + s]
            out = out.type_as(x)
            out = rearrange(out, 'b d l1 l2 -> b d (l1 l2)')
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
        return 'edim={}, ndim={}, bidirectional={}, trunction={}'.format(self.embed_dim, self.ndim,
                                                                         self.bidirectional,
                                                                         self.truncation)


def run_steps(ssm2d, x, residual_and_silu=False):
    # X shape is l,l,B,embed_dim
    L = x.shape[0]
    l = int(math.sqrt(L))
    B = x.shape[1]
    D = x.shape[2]
    # x = x.permute(1, 2, 0)  # L, B ,D -> B, D, L
    x = x.view(l, l, B, D, 1)
    orig_x = x.clone().squeeze(-1) * ssm2d.omega
    results = torch.zeros_like(x).squeeze(-1)
    states_h = torch.zeros(l, l, B, ssm2d.embed_dim, ssm2d.ndim)  # L, B, embed_dim, n_dim
    states_v = torch.zeros(l, l, B, ssm2d.embed_dim, ssm2d.ndim)  # L, B, embed_dim, n_dim
    for i in range(x.size(0)):
        for j in range(x.size(1)):
            state_h_left = states_h[i, j - 1] if j - 1 >= 0 else None
            state_v_left = states_v[i, j - 1] if j - 1 >= 0 else None
            state_h_up = states_h[i - 1, j] if i - 1 >= 0 else None
            state_v_up = states_v[i - 1, j] if i - 1 >= 0 else None
            y, state_h, state_v = ssm2d.one_step(x[i, j], state_h_left, state_v_left, state_h_up, state_v_up)
            results[i, j] = y
            states_h[i, j] = state_h
            states_v[i, j] = state_v

    if residual_and_silu:
        results += orig_x
        results = F.silu(results)

    return results, states_h


def run_steps_bidirectional(ssm2d, x, residual_and_silu=False, device='cpu'):
    # X shape is l,l,B,embed_dim
    L = x.shape[0]
    l = int(math.sqrt(L))
    B = x.shape[1]
    D = x.shape[2]
    # x = x.permute(1, 2, 0)  # L, B ,D -> B, D, L
    x = x.view(l, l, B, D, 1)
    orig_x = (x.clone().squeeze(-1) * ssm2d.omega).to(device)
    flips = [[], [0], [1], [0, 1]]
    x_instances = [torch.flip(x.clone(), dims=flip) for flip in flips]
    final_results = torch.zeros_like(x).squeeze(-1).to(device)
    for idx, instance in enumerate(x_instances):
        results = torch.zeros_like(x).squeeze(-1).to(device)
        states_h = torch.zeros(l, l, B, ssm2d.embed_dim, ssm2d.ndim).to(device)  # L, B, embed_dim, n_dim
        states_v = torch.zeros(l, l, B, ssm2d.embed_dim, ssm2d.ndim).to(device)  # L, B, embed_dim, n_dim
        for i in range(instance.size(0)):
            for j in range(instance.size(1)):
                state_h_left = states_h[i, j - 1] if j - 1 >= 0 else None
                state_v_left = states_v[i, j - 1] if j - 1 >= 0 else None
                state_h_up = states_h[i - 1, j] if i - 1 >= 0 else None
                state_v_up = states_v[i - 1, j] if i - 1 >= 0 else None
                y, state_h, state_v = ssm2d.one_step(instance[i, j], state_h_left, state_v_left, state_h_up,
                                                     state_v_up,
                                                     bidirectional_index=idx)
                results[i, j] = y
                # print('idx instance', idx, 'i j', i, j, 'y results', y.squeeze(0))
                states_h[i, j] = state_h
                states_v[i, j] = state_v
        final_results += torch.flip(results, dims=flips[idx])
        # print('step results', results.squeeze(-1).squeeze(-1))

    if residual_and_silu:
        final_results += orig_x
        final_results = F.silu(final_results)

    return final_results, None


def test_ema():
    device = 'cpu'
    ndim = 2
    embed_dim = 3
    L = 2 ** 2
    random_x = False
    bidirectional = False
    # truncation = None
    seed = 42
    torch.manual_seed(seed)
    ssm2d = TwoDimensionalSSM(embed_dim, ndim, bidirectional, L=L)
    ssm2d.to(device)

    # X creation
    B = 1
    if random_x:
        x = torch.randn(L, B, embed_dim).to(device)
    else:
        x = (torch.arange(L, dtype=torch.float) + 1).view(L, 1, 1).repeat(1, B, embed_dim).to(device)
    conv_y = ssm2d(x)

    if bidirectional:
        results_step, states_step = run_steps_bidirectional(ssm2d, x, True, device)
    else:
        results_step, states_step = run_steps(ssm2d, x, True)

    results_step = results_step.view(L, B, embed_dim)
    print('The max difference is:', torch.norm(conv_y - results_step).max())
    print('The difference is:', (conv_y - results_step).squeeze(-1).squeeze(-1))
    assert torch.allclose(conv_y, results_step, atol=1e-4)


def test_kernel_to_sympy():
    device = 'cpu'
    ndim = 1
    embed_dim = 1
    L = 32 ** 2
    L_one_sided = int(math.sqrt(L))

    random_x = False
    bidirectional = False
    # truncation = None
    seed = 42
    torch.manual_seed(seed)
    ssm2d = TwoDimensionalSSM(embed_dim, ndim, bidirectional, L=L, force_coeff_calc=True)
    ssm2d.to(device)

    # X creation
    B = 1
    if random_x:
        x = torch.randn(L, B, embed_dim).to(device)
    else:
        x = (torch.arange(L, dtype=torch.float) + 1).view(L, 1, 1).repeat(1, B, embed_dim).to(device)
    sympy_kernel = ssm2d.compute_sympy_kernel()
    kernel = ssm2d._compute_kernel(L_one_sided).squeeze(-1)
    sympy_kernel = torch.from_numpy(sympy_kernel.values.astype(np.float32)).to(device)
    print(kernel - sympy_kernel)
    assert torch.allclose(kernel, sympy_kernel, atol=1e-4)


if __name__ == '__main__':
    test_kernel_to_sympy()
