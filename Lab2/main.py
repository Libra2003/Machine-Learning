# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
x_train = np.array([1.0,2.0])
y_train = np.array([200.0,300.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")