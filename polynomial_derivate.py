import numpy as np
import matplotlib.pyplot as plt


def derAtPoint(f, x):
    h = 0.00001
    return (f(x+h)-f(x))/h


def derivative(f, degree, points, precision):
    x = [x / precision for x in range(-points//2, points//2)]
    y = [derAtPoint(f, i) for i in x]
    plt.plot(x, y, label="derivative")
    plt.plot(x, [f(i) for i in x], label="function")
    plt.legend()
    plt.show()
    dimensions = degree

    powers = np.zeros((dimensions, dimensions), dtype=float)
    for i in range(dimensions-1, -1, -1):
        for j in range(0, dimensions):
            powers[dimensions-1-j][i] = dimensions-1-i+j

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
            b[dimensions-i-1] = [total]

    B = np.matrix(b)

    X = np.linalg.pinv(A) * B

    coefficients = list(np.ndarray.flatten(np.asarray(X)))

    output_string = []
    for i in range(len(coefficients)):
        if abs(coefficients[i]) < 1e-2:
            coefficients[i] = 0
        string = ""
        string += str(coefficients[i])
        if i != len(coefficients) - 1:
            string += "*x**" + str(len(coefficients)-i-1)
        output_string.append(string)
    output_string = " + ".join(output_string)
    print(output_string)

    return string_to_func(output_string)


def string_to_func(string):
    def g(x):
        new_str = eval(string.replace('x', f"({str(x)})"))
        return new_str
    return g


def f(x):
    import math
    return math.exp(x)


starting_degree = 5
for i in range(starting_degree, 0, -1):
    result = derivative(f, i, 200, 200)
    f = result
