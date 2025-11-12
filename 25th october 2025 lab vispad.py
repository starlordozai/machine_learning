def l1_norm(v):
    total = 0
    for x in v:
        total += abs(x)
    return total

v = [45, -4]
print(f"l1_norm = {l1_norm(v)}")


def euclidean_distance(a, b):
    total = 0
    for i in range(len(a)):
        total += (a[i] - b[i])**2
    return (total)**0.5   # Correct square root

a = [7, 9]
b = [4, 0]
print("Distance between", a, "and", b, "=", euclidean_distance(a, b))

import numpy as np

def dot_product(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2)   # or v1 @ v2

v1 = [1, 2, 3]
v2 = [4, 5, 6]

print("Using NumPy:", dot_product(v1, v2))  # Expected: 32
