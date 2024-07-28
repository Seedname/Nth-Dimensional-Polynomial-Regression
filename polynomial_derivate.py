import numpy as np
import sys
import matplotlib.pyplot as plt
import decimal
import mpmath as mp

mp.dps = 100

def derAtPoint(f, x):
    h =
    return (f(x+h)-f(x))/h



def derivative(f, degree, precision):
    x = list(range(-precision//2, precision//2))

    y = [derAtPoint(f, i) for i in x]
    plt.plot(x, [f(i) for i in x])
    plt.plot(x, y)
    plt.show()
    dimensions = degree

    coefficients = fit_function(dimensions, x, y)

    output_string = ' + '.join([f"{coefficients[i]:.4f}*x**{len(coefficients)-i-1}"
                                for i in range(len(coefficients))])

    return output_string


def fit_function(dimensions, x, y):
    powers = np.zeros((dimensions, dimensions), dtype=float)
    for i in range(dimensions - 1, -1, -1):
        for j in range(0, dimensions):
            powers[dimensions - 1 - j][i] = dimensions - 1 - i + j
    a = np.zeros((dimensions, dimensions), dtype=float)
    for i in range(dimensions):
        for j in range(dimensions):
            total = 0
            for v in x:
                total += pow(v, powers[i][j])
            a[i][j] = total
    A = np.matrix(a)
    b = [0] * dimensions
    for i in range(dimensions):
        total = 0
        for j in range(len(x)):
            total += pow(x[j], i) * y[j]
            b[dimensions - i - 1] = [total]
    B = np.matrix(b)
    X = np.linalg.pinv(A) * B
    coefficients = list(np.ndarray.flatten(np.asarray(X)))
    return coefficients


def string_to_func(string, x):
    new_str = eval(string.replace('x', str(x)))
    return new_str



def main() -> None:
    def f(x):
        return x ** 2 + x ** 3

    current_string = derivative(f, 2, 2000)

    def g(x):
        return string_to_func(current_string, x)

    starting_degree = 3
    for i in range(starting_degree, 0, -1):
        current_string = derivative(f, i, 2000)
        f = g
        print(current_string)

if __name__ == "__main__":
    main()
