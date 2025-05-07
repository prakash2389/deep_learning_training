import numpy as np

# Simple quadratic function: y = x^2
def function(x):
    return x**2

# Gradient of the function: dy/dx = 2x
def gradient(x):
    return 2 * x

# Gradient descent parameters
learning_rate = 0.1
iterations = 100
x = 10  # Starting point

# Perform gradient descent
for i in range(iterations):
    grad = gradient(x)
    x = x - learning_rate * grad
    print(f"Iteration {i+1}: x = {x}, f(x) = {function(x)}")
