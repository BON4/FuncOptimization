from scipy import linalg
import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import LinearConstraint
from matplotlib import cm
from tabulate import tabulate

START_POINT = [1, 0]


def fun(x: np.ndarray) -> np.float:
    # return x[0]**4+x[1]**4
    return (np.exp(x[0]) + np.exp(x[1]))


def hfun(x: np.ndarray) -> np.float:
    return (x[0]**2+x[1]**2-9)


def gfun1(x: np.ndarray) -> np.float:
    # return (6*x[0]-x[1]**2-6)
    return (x[0]+x[1]-1)


def gfun2(x: np.ndarray) -> np.float:
    return x[0]


def gfun3(x: np.ndarray) -> np.float:
    return x[1]


def gplusfun(g, x, u, c):
    return min(g(x), np.divide(u, c))


def modified_lagrange_method(fun, points, epsx, g_constarins, h_constrains, u=[0], a=[0], c=0.1, beta=2, counter=0):
    """Minimize a function with given constrains

    Arguments:
        points {[float]} -- [array of calculated points]
        epsx {[float]} -- [epsilon]
        g_constarins {[Callable]} -- [array of inequality constrains]
        h_constrains {[Callable]} -- [array of equality constrains]

    Keyword Arguments:
        u {list} -- [Langrange factor for inequality] (default: {[0]})
        a {list} -- [Langrange factor for equality] (default: {[0]})
        c {float} -- [penalty factor] (default: {0.1})
        beta {int} -- [growth rate of penalty factor must be in range [2;26]] (default: {2})
        counter {int} -- [counter] (default: {0})
    """

    def lagrange(x):
        if(len(g_constarins) != 0 and len(g_constarins) != 0):
            array_of_constrains_g = np.array(
                [gplusfun(g_constrain, x, u_i, c) for g_constrain, u_i in zip(g_constarins, u)])
            array_of_constrains_h = np.array(
                [h_constrain(x) for h_constrain in h_constrains])
            return fun(x) - sum([u_i * g for u_i, g in zip(u, array_of_constrains_g)]) + 0.5*sum(c * array_of_constrains_g**2) - sum([a_i * g for a_i, g in zip(a, array_of_constrains_h)]) + 0.5*sum(c * array_of_constrains_h**2)
        elif(len(h_constarins) != 0 and len(g_constarins) == 0):
            array_of_constrains_h = np.array(
                [h_constrain(x) for h_constrain in h_constrains])
            return fun(x) - sum([a_i * h for a_i, h in zip(a, array_of_constrains_h)]) + 0.5*sum(c * array_of_constrains_h**2)
        elif(len(h_constarins) == 0 and len(g_constarins) != 0):
            array_of_constrains_g = np.array(
                [gplusfun(g_constrain, x, u_i, c) for g_constrain, u_i in zip(g_constarins, u)])
            return fun(x) - sum([u_i * g for u_i, g in zip(u, array_of_constrains_g)]) + 0.5*sum(c * array_of_constrains_g**2)
        else:
            return fun(x)

    next_val = sopt.minimize(
        lagrange, x0=points[-1], method='BFGS').x

    points.append(next_val)

    u = np.array([max(0, (u_i - c*g_constrain(next_val)))
                  for g_constrain, u_i in zip(g_constarins, u)])

    a = np.array([a_i-c*h_constrain(next_val)
                  for h_constrain, a_i in zip(h_constrains, a)])

    c = beta*c

    counter = counter+1

    if(abs(next_val - points[-2])[0] < epsx and abs(next_val - points[-2])[1] < epsx):
        return points
    else:
        return modified_lagrange_method(fun, points, epsx, g_constarins, h_constrains, u, a, c, beta, counter)


def plotting():

    X, Y = np.mgrid[-2: 3: 20j, -4.5: 3: 20j]
    Z = fun(np.array([X, Y]))

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot_wireframe(X, Y, Z)

    ax.contour(X, Y, fun(
        np.array([X, Y]))-gfun1(np.array([X, Y])), levels=[14], colors='green')

    X, Z = np.mgrid[-2: 3: 20j, 0: 20: 20j]
    Y = (lambda x: np.sqrt(
        9-x[0]**2))(np.array([X, Y]))

    ax.plot_wireframe(X, -Y, Z)

    g_constarins = np.array([gfun1, gfun2, gfun3])
    h_constrains = np.array([hfun])

    vals = modified_lagrange_method(fun, list([START_POINT, ]), 0.001,
                                    g_constarins, h_constrains)

    ax.plot(np.array(vals).T[0], np.array(vals).T[1], np.array(
        list(map(fun, np.array(vals)))).T, "x-", color='red')

    plt.show()


plotting()
