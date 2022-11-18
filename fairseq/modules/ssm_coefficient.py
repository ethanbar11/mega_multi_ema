import os

import torch
from sympy import *


# Define signal matrix u which is ([-1,L], [-1,L])


def convert_coeffs_dict_to_vec(coeff_dict, L):
    max_pow = L * 2
    vec = torch.zeros(max_pow * max_pow * 2)
    for key, val in coeff_dict.items():
        # If doesn't have any values inside
        if key == 0:
            continue
        index = 0
        if str(key) == 'B_2':
            index += L * 2 * max_pow
        else:
            for arg in key.args:
                if len(arg.args) == 0:
                    s = str(arg)
                    pow = 1
                else:
                    s = str(arg.args[0])
                    pow = int(arg.args[1])
                if s == 'A_1':
                    index += max_pow * pow
                elif s == 'A_2':
                    index += pow
                elif s == 'B_2':
                    index += L * 2 * max_pow

        print(key, index)
        vec[index] = int(val)
    return vec


class CoeffCalculator:
    def __init__(self, L):
        self.L = L
        print(self.L)
        self.u = []

        self.u_0 = Symbol('u_0_0')
        self.u_0 = 1

        self.A_1 = Symbol('A_1')
        self.A_2 = Symbol('A_2')
        self.A_3 = Symbol('A_3')
        self.A_4 = Symbol('A_4')
        self.B_1 = Symbol('B_1')
        self.B_2 = Symbol('B_2')

        self.A_4 = self.A_1
        self.A_3 = self.A_2

        self.C_1 = Symbol('C_1')
        self.C_2 = Symbol('C_2')

        # Define that where i=-1 or j=-1 x[i,j] = 0
        self.x_h = [[0 for i in range(self.L)] for j in range(self.L)]
        self.x_v = [[0 for i in range(self.L)] for j in range(self.L)]

        self.final_coeffs_matrix_horizontal = torch.zeros(L * L, L * L * 8)
        self.final_coeffs_matrix_vertical = torch.zeros(L * L, L * L * 8)
        # Print files in current directory
        self.horizontal_cache_location = '/home/ethan_baron/mega/coeffs_cache/horizontal.pt'
        self.vertical_cache_location = '/home/ethan_baron/mega/coeffs_cache/vertical.pt'

    def calc_coeffs_lazy(self):
        if os.path.exists(self.horizontal_cache_location) and os.path.exists(self.vertical_cache_location):
            horizontal = torch.load(self.horizontal_cache_location)
            if horizontal.shape[0] == self.L * self.L:
                self.final_coeffs_matrix_horizontal = horizontal.type(torch.DoubleTensor)
                self.final_coeffs_matrix_vertical = torch.load(self.vertical_cache_location).type(torch.DoubleTensor)
                return
        self.initialize_matrices()
        self.set_final_coeffs_matrix()
        torch.save(self.final_coeffs_matrix_horizontal, self.horizontal_cache_location)
        torch.save(self.final_coeffs_matrix_vertical, self.vertical_cache_location)

    def initialize_matrices(self):
        for i in range(self.L):
            for j in range(self.L):
                self.x_h[i][j] = self.calc_x_h(i, j)
                self.x_v[i][j] = self.calc_x_v(i, j)

    # Define expression with A_1 .. A_4, B_1,B_2 as variables
    def calc_x_h(self, i, j):
        last_j = j - 1
        relevant_x_h = self.x_h[i][last_j] if i >= 0 and last_j >= 0 else 0
        relevant_x_v = self.x_v[i][last_j] if i >= 0 and last_j >= 0 else 0
        current_u = self.u_0 if i == 0 and j == 0 else 0
        expr = self.A_1 * relevant_x_h + self.A_2 * relevant_x_v + self.B_1 * current_u
        return expand(expr)

    def calc_x_v(self, i, j):
        last_i = i - 1
        relevant_x_h = self.x_h[last_i][j] if i >= 0 and last_i >= 0 else 0
        relevant_x_v = self.x_v[last_i][j] if i >= 0 and last_i >= 0 else 0
        current_u = self.u_0 if i == 0 and j == 0 else 0
        expr = self.A_3 * relevant_x_h + self.A_4 * relevant_x_v + self.B_2 * current_u
        return expand(expr)

    def set_final_coeffs_matrix(self):
        for i in range(self.final_coeffs_matrix_horizontal.shape[0]):
            ver = i // self.L
            hor = i % self.L
            print('location:', ver, hor)
            self.final_coeffs_matrix_horizontal[i] = convert_coeffs_dict_to_vec(
                self.x_h[ver][hor].as_coefficients_dict(), self.L)
            self.final_coeffs_matrix_vertical[i] = convert_coeffs_dict_to_vec(
                self.x_v[ver][hor].as_coefficients_dict(), self.L)
        self.final_coeffs_matrix_horizontal = self.final_coeffs_matrix_horizontal.type(torch.DoubleTensor)
        self.final_coeffs_matrix_vertical = self.final_coeffs_matrix_vertical.type(torch.DoubleTensor)


if __name__ == '__main__':
    L = 32
    calc = CoeffCalculator(L)
    calc.calc_coeffs_lazy()
    print(calc.final_coeffs_matrix_horizontal)
    print(calc.final_coeffs_matrix_vertical)
