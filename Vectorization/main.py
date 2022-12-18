import numpy as np

import time

w = np.array([1.0,2.5,3.3])
b = 4
x = np.array([10,20,30])
start = time.time_ns()
time.sleep(1)
f = (w[0]*x[0]+w[1]*x[1]+w[2]*x[2]) +b

print(f)
end = time.time_ns()
print("Time 1: ", end - start)

start1 = time.time_ns()
time.sleep(1)
f = 0
for j in range(3):
    f = f + w[j]*x[j]

f = f+b

print(f)
end1 = time.time_ns()
print("Time 2: ", end1 - start1)

start3 = time.time_ns()
time.sleep(1)
f = np.dot(w,x)+b
print(f)
end3 = time.time_ns()

print("Time 3: ", end3 - start3)

