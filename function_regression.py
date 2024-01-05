import math
import numpy as np
import matplotlib.pyplot as plt

eps = np.finfo(float).eps

def nthDerivative(f, n, x):
    sum = 0
    dx = 0.0001
    for k in range(n+1):
        sum += pow(-1, k)*(math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))*((f(x+dx*(n-k)))/pow(dx, n))
    return sum

def taylor(f, a, x, u):
    sum = 0
    for n in range(u+1):
        sum += pow(x-a, n) * nthDerivative(f, n, a) / math.factorial(n)
    return sum

def f(x):
    return x ** 2

def g(x):
    return taylor(f,1,x,6)

x = list(range(0,100))
y = list(map(g, x))

plt.plot(x, y)
plt.show()