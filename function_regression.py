import math
import numpy as np

eps = np.finfo(float).eps

def nth_derivative(f, x, n):
    total = 0
    # h = eps * math.sqrt(abs(x) + math.sqrt(eps))
    h = 0.01

    for k in range(n+1):
        total += pow(-1, k)*(math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))*((f(x+h*(n-k)))/pow(h, n))

    return total

def taylor(f, a, x, u):
    total = 0
    for n in range(u+1):
        total += pow(x-a, n) * nth_derivative(f, a, n) / math.factorial(n)
    return total

def f(x):
    return x ** 2   

# 1/2 * x ^ -1/2
# print(1/2 * 5 ** -1/2)
print(taylor(f, 10, 5, 10))