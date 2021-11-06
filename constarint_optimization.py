from scipy import linalg
import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import LinearConstraint
from matplotlib import cm
import tabulate
import math

START_POINT = [2,2]

# fun - main muction
def fun(x: np.ndarray) -> np.float64:
    return ((x[0]-2)**2) + (x[1]-1)**2

# hfunc - represents an "= 0" fun
def hfun(x: np.ndarray) -> np.float64:
    return x[0]-2*x[1]+1

# gfunc - represents an ">= 0" fun
def gfun(x: np.ndarray) -> np.float64:
    return -0.25 * x[0]**2 - x[1]**2+1


def gplusfun(g, x, u, c):
    return min(g(x), np.divide(u, c))


def modified_lagrange_method(fun, points, epsx, g_constarins, h_constrains, u=[0], a=[0], c=0.1, beta=2, counter=0, func_counter = 0, _callback=None):
    """Minimize a function with given constrains
    Arguments:
        points {[float]} -- [array of calculated points]
        epsx {[float]} -- [epsilon]
        g_constarins {[Callable]} -- [array of inequality constrains]
        h_constrains {[Callable]} -- [array of equality constrains]
    Keyword Arguments:
        u {list} -- [Langrange factor for inequality, must be same length as g_constarins] (default: {[0]})
        a {list} -- [Langrange factor for equality, must be same length as h_constrains] (default: {[0]})
        c {float} -- [penalty factor] (default: {0.1})
        beta {int} -- [growth rate of penalty factor must be in range [2;26]] (default: {2})
        counter {int} -- [counter] (default: {0})
        callback - function that takes dict x, witch contains all intermediate values such as x, u, a, c, f(x), L(x,u,a,c)
    """
    def lagrange(x):
        if(len(g_constarins) != 0 and len(g_constarins) != 0):
            array_of_constrains_g = np.array(
                [gplusfun(g_constrain, x, u_i, c) for g_constrain, u_i in zip(g_constarins, u)])
            array_of_constrains_h = np.array(
                [h_constrain(x) for h_constrain in h_constrains])
            return fun(x) - sum([u_i * g for u_i, g in zip(u, array_of_constrains_g)]) + 0.5*sum(c * array_of_constrains_g**2) - sum([a_i * g for a_i, g in zip(a, array_of_constrains_h)]) + 0.5*sum(c * array_of_constrains_h**2)
        elif(len(h_constrains) != 0 and len(g_constarins) == 0):
            array_of_constrains_h = np.array(
                [h_constrain(x) for h_constrain in h_constrains])
            return fun(x) - sum([a_i * h for a_i, h in zip(a, array_of_constrains_h)]) + 0.5*sum(c * array_of_constrains_h**2)
        elif(len(h_constrains) == 0 and len(g_constarins) != 0):
            array_of_constrains_g = np.array(
                [gplusfun(g_constrain, x, u_i, c) for g_constrain, u_i in zip(g_constarins, u)])
            return fun(x) - sum([u_i * g for u_i, g in zip(u, array_of_constrains_g)]) + 0.5*sum(c * array_of_constrains_g**2)
        else:
            return fun(x)
    
    if _callback is not None:
        _callback({"x": points[-1], "u": u, "a": a, "c": c, "f": fun(points[-1]), "L": lagrange(points[-1]), "iter": counter, "fiter": func_counter})

    # BFGS - is most fast & eficient for my cases
    res = sopt.minimize(
        lagrange, x0=points[-1], method='BFGS')

    next_val = res.x
    func_counter = func_counter+res.nfev
    counter = counter+res.nit
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
        return modified_lagrange_method(fun, points, epsx, g_constarins, h_constrains, u, a, c, beta, counter, func_counter,_callback)


def filter_zLim(X,Y,Z, zlim):
    for i in range(0, len(Z)):
        for j in range(0, len(Z)):
            if Z[i][j] > zlim[1] or Z[i][j] < zlim[0]:
                Z[i][j] = 4

    return X, Y, Z

def printDecorator(f, res):
    def wrapper(x):
        res.append(x)
        ret_val = f(x)
        return ret_val
    return wrapper

def trunc(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def plotting():
    xs = []
    results_list = []
    f = lambda x: xs.append(x["x"])
    callback = printDecorator(f, results_list)

    #Adjust plotting scale here where x in [a1,b1] and y in [a2,b2] [a1: b1: 20j, a2: b2: 20j]
    X, Y = np.mgrid[2.4:0:20j, 2.5:-1.5:20j]
    Z = fun(np.array([X, Y]))

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlim((0, 3))
    ax.set_ylim((-1.5, 3))
    ax.set_zlim((0, 4))

    ax.plot_wireframe(X, Y, Z)

    ax.contour(X, Y, gfun(np.array([X, Y])), levels=[0], colors='blue')
    ax.contour(X, Y, hfun(np.array([X, Y])), levels=[0], colors='lime')

    #Put list of constrains here, for my case its one constrain g(x) and one h(x)
    g_constarins = np.array([gfun])
    h_constrains = np.array([hfun])

    vals = modified_lagrange_method(fun, list([START_POINT, ]), 1e-6,
                                    g_constarins, h_constrains, _callback=callback)

    #Print Results Table
    header = results_list[0].keys()
    rows =  [x.values() for x in results_list[0:11] + [results_list[-1]]]
    print(tabulate.tabulate(rows, header, tablefmt='grid'))

    
    ax.plot(np.array(vals).T[0], np.array(vals).T[1], np.array(
        list(map(fun, np.array(vals)))).T, "x-", color='red')

    ax1 = fig.add_subplot(1, 2, 2)

    #Adjust plotting scale here where x in [a1,b1] and y in [a2,b2] [a1: b1: 20j, a2: b2: 20j]
    X, Y = np.mgrid[3: 0: 20j, 2.3: -1.5: 20j]
    Z = fun(np.array([X, Y]))

    ax1.contour(X, Y, Z, levels=40)

    t = 0
    for x in zip(np.array(vals).T[0], np.array(vals).T[1]):
        if abs(fun(x) - t) > 1e-2:
            ax1.annotate(trunc(fun(x),3), (x[0], x[1]))
        t = fun(x)

    ax1.plot(np.array(vals).T[0], np.array(vals).T[1], "x-", color='red')

    for idx, g_constr in enumerate(g_constarins):
        ax1.clabel(ax1.contour(X, Y, g_constr(np.array([X, Y])), levels=[0], colors='blue'), fmt=f"g{idx}(x)", fontsize=10)

    for idx, h_constr in enumerate(h_constrains):
        ax1.clabel(ax1.contour(X, Y, h_constr(np.array([X, Y])), levels=[0], colors='lime'), fmt=f"h{idx}(x)", fontsize=10)


    plt.show()

plotting()
