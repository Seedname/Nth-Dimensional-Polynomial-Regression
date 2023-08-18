import numpy as np

def derAtPoint(f, x):
    h = 0.0000001
    return (f(x+h)-f(x))/h

def derivative(f, degree, precision):
    x = list(range(-precision//2, precision//2))
    y = [derAtPoint(f, i) for i in x]

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

    return ' + '.join([f"{str('%.4f' % round(coefficients[i], 4)).rstrip('0')}*x^{len(coefficients)-i-1}" for i in range(len(coefficients)) if round(coefficients[i], 3) != 0])

def f(x):
    return x**2 * (x**3 + 7 + x**2 + 8*x) + 3.05*x

print(derivative(f, 5, 20))