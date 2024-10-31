#


import math
import numpy as np
import matplotlib.pyplot as plt

def taylor(f, a, x, u):
    sum = 0
    for n in range(u+1):
        sum += (x-a)**n * f(a, n) / math.factorial(n)
    return sum

def f(x, n=0):
    if n == 0:
        return x**2
    elif n == 1:
        return 2*x
    elif n == 2:
        return 2
    else:
        return 0

def g(x):
    return taylor(f, 1, x, 2)

x = np.linspace(0, 100, 100)  # Changed range to 0-2
y_taylor = [g(xi) for xi in x]
y_actual = [f(xi) for xi in x]

plt.plot(x, y_taylor, label='Taylor approximation')
plt.plot(x, y_actual, label='Actual function')
plt.legend()
plt.title('Taylor approximation of f(x) = x^2 around x=1')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
