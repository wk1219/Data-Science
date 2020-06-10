import numpy as np

def numeriacal_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

# init_x : initial value, lr : learning rate, step_num : repeat count
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numeriacal_gradient(f, x)
        x -= lr * grad
    return x

def function(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
res0 = gradient_descent(function, init_x=init_x, lr=0.1, step_num=100)
print(res0)

res1 = gradient_descent(function, init_x=init_x, lr=10.0, step_num=100)
print(res1)

res2 = gradient_descent(function, init_x=init_x, lr=1e-10, step_num=100)
print(res2)