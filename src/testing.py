import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10, 11, 12, 13, 15, 16])

len_diff = abs(len(a) - len(b))
if len(a) < len(b):
    a = np.concatenate((a, b[len(b) - len_diff :]))

print(a)
print(b)
