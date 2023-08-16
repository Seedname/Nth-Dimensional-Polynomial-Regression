import numpy as np
import matplotlib.pyplot as plt


x = [-0.452, 0, 0.322, -0.931]
y = [0, 1, 2.548, -2.361]

dimensions = 4

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
print(coefficients)

x_points = np.linspace(min(x)-1, max(x)+1, 100)
y_poly = np.polyval(coefficients, x_points)

plt.figure()

plt.plot(x_points, y_poly, label='Polynomial Function', color='red')

for i in range(len(x)):
    plt.scatter(x[i], y[i], color='blue')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of Points and Polynomial Function')
plt.legend()

plt.show()