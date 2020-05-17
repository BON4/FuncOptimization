# Метод ДСК 4
# Функция 4
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from tabulate import tabulate

answer_table = pd.DataFrame(
    columns=['x0', 'x1', 'x2', 'x3', 'f(x1)', 'f(x2)', 'f(x3)', 'Xmin'])

array = list()


def fun(x: np.float) -> np.float:
    """
    Put your function here
    """
    return 1-1-np.power(np.power(((x-6)/5), 4), 0.25)


def svenn(x0, delta):
    print("Starting Svenn algorithm...")
    x1 = x0
    x2 = x1+delta
    if fun(x1) > fun(x2):
        delta = -delta
        x1 = x0
        x2 = x1+delta
    while fun(x2) > fun(x1):
        delta *= 2
        x1 = x2
        x2 = x1+delta
        print("x1: " + str(x1) + " , f(x1): " + str(fun(x1)))
        print("x2: " + str(x2) + " , f(x2): " + str(fun(x2)))

    a = x2 - 3*delta/2
    b = x2 - delta/2

    if a > b:
        print("a: " + str(a))
        print("b: " + str(b))
    else:
        temp = b
        b = a
        a = temp
        print("a: " + str(a))
        print("b: " + str(b))
    return [a, b]


def dsc(x0, delta, epsx, counter=0):
    global array
    global answer_table
    counter += 1
    print(f"Iteartion {counter}...")
    svenn_res = svenn(x0, delta)

    a = svenn_res[0]
    b = svenn_res[1]

    x1 = a
    x3 = b
    x2 = (abs(b) + abs(a))/2

    print("x1 = %8.5f, x2 = %8.5f, x3 = %8.5f" % (x1, x2, x3))

    xs = x2 + ((x3-x2)*(fun(x1) - fun(x3)))/(2*(fun(x1) - 2*fun(x2) + fun(x3)))

    answer_table = answer_table.append(
        {'x0': x0, 'x1': x1, 'x2': x2, 'x3': x3, 'f(x1)': fun(x1), 'f(x2)': fun(x2), 'f(x3)': fun(x3), 'Xmin': xs}, ignore_index=True)

    plt.scatter(a, fun(a), 20, color='red')
    plt.scatter(b, fun(b), 20, color='red')
    plt.scatter(xs, fun(xs), 20, color='green')

    if(abs(xs - x0) < epsx):
        return [a, b, xs]
    else:
        return dsc(xs, delta, epsx, counter)


a, b, xs = dsc(2, 0.2, 0.001)

print(f"Xmin = {xs}, a = {a}, b = {b}")

print(tabulate(answer_table, headers='keys', tablefmt='psql'))

x = list(map(fun, np.arange(1, 11, 0.01)))
plt.plot(list(np.arange(1, 11, 0.01)), x)
plt.grid(True)
plt.show()
