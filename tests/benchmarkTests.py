# ######
# ######  PURE CPU
# ######
# import numpy as np
# from timeit import default_timer as timer
#
# def pow(a, b, c):
#     for i in range(a.size):
#          c[i] = a[i] ** b[i]
#
# def main():
#     vec_size = 100000000
#
#     a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
#     c = np.zeros(vec_size, dtype=np.float32)
#
#     start = timer()
#     pow(a, b, c)
#     duration = timer() - start
#
#     print(duration)  # 55.19145088028804
#
# if __name__ == '__main__':
#     main()


# ######
# ######  CPU with Threads ?? (numba library)
# ######
# import numpy as np
# from timeit import default_timer as timer
# from numba import jit
#
# @jit(nopython=True, parallel=True)
# def pow(a, b, c):
#     for i in range(a.size):
#          c[i] = a[i] ** b[i]
#
# def main():
#     vec_size = 100000000
#
#     a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
#     c = np.zeros(vec_size, dtype=np.float32)
#
#     start = timer()
#     pow(a, b, c)
#     duration = timer() - start
#
#     print(duration) # 1.5194660072366057
#
# if __name__ == '__main__':
#     main()


# #####
# #####  GPU
# #####
# import numpy as np
# from timeit import default_timer as timer
# from numba import vectorize
#
# @vectorize(['float32(float32, float32)'], target='cuda')
# def pow(a, b):
#     return a ** b
#
# def main():
#     vec_size = 100000000
#
#     a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
#     c = np.zeros(vec_size, dtype=np.float32)
#
#     start = timer()
#     c = pow(a, b)
#     duration = timer() - start
#
#     print(duration)  # 1.4808983078073659
#
# if __name__ == '__main__':
#     main()