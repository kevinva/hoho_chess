
from numba import jit, int32
import numpy as np
import time


# @jit(nopython=True)
# def add(x, y):
#     return x + y

# # 相当于类型检查
# @jit(numba.int32(numba.int32, numba.int32))
# def add_signatured(x, y):
#     return x + y

# # print(f'2+3={add_signatured(2, 3)}')
# # print(f'2.1+3.5={add_signatured(2.1, 3.5)}')
# # print(f'2j+2={add_signatured(2j, 3)}')

# print(f'2+3={add(2, 3)}')
# print(f'2.1+3.5={add(2.1, 3.5)}')
# print(f'2j+2={add(2j, 3)}')

SIZE = 2000
x = np.random.random((SIZE, SIZE))

@jit("int32(int32, int32)", nopython=True)
def f2(x, y):
    return x + y

@jit
def jit_tan_sum(a):
    tan_sum = 0
    for i in range(SIZE):
        for j in range(SIZE):
            tan_sum += np.tanh(a[i, j])
    return tan_sum

start = time.time()
jit_tan_sum(x)
end = time.time()
print(f'Elapsed (with compilation) = {end - start}')

start = time.time()
jit_tan_sum(x)
end = time.time()
print(f'Elapsed (with compilation) = {end - start}')