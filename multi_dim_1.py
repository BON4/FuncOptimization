import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as sopt
import scipy.optimize as opt


def fun(x):
    # return 0.5*x[0]**2 + 2.5*x[1]**2
    # return (x[0]**2 + x[1]**2)
    return (np.power((np.power(x[1], 2) + x[0] - 11), 2)+np.power((x[0]+np.power(x[1], 2)-7), 2))


def dfun(x):
    # return np.array([x[0], 5*x[1]])
    # return np.array([2*x[0], 2*x[1]])
    return np.array([(4*x[0] + 4*np.power(x[1], 2)-36), (8*np.power(x[1], 3)+8*x[0]*x[1]-72*x[1])])


def hesfun(x):

    H = [
        [4, 8*x[1]],
        [8*x[1], 24*(x[1]**2) + 8*x[0]-72]
    ]

    return np.array(H)


def line_search(s, f, H):
    return np.dot(-s.T, f)/np.dot(np.dot(s.T, H), s)


def polack_ribier_ethod(poits: list, max_iters: int, s: np.ndarray = np.nan, alpha_opt: np.float = np.nan,
                        counter=1) -> np.ndarray:
    def f1d(alpha):
        return fun(x + alpha*s)

    x = poits[-1]

    if(counter <= 1):
        f = dfun(x)
        s = -dfun(x)
        alpha_opt = line_search(s, f, hesfun(x))
        next_guess = x + alpha_opt * s
        poits.append(next_guess)
    else:
        f = dfun(x)
        f_old = dfun(poits[-1])
        w = ((f-(f_old)).T*f)/(f_old.T*f_old)
        s = -f + w * s
        alpha_opt = line_search(s, f, hesfun(x))
        next_guess = x + alpha_opt * s
        poits.append(next_guess)

    if(counter > max_iters):
        return poits
    return polack_ribier_ethod(poits, max_iters, s, alpha_opt, counter+1)


def steepest_descent(poits: list, max_iters: int, s: np.ndarray = np.nan, alpha_opt: np.float = np.nan,
                     counter=1) -> np.ndarray:
    def f1d(alpha):
        return fun(x + alpha*s)
    x = poits[-1]
    s = -dfun(x)
    alpha_opt = sopt.golden(f1d)
    next_guess = x + alpha_opt * s
    poits.append(next_guess)

    if(counter > max_iters):
        return poits
    return steepest_descent(poits, max_iters, s, alpha_opt, counter+1)


def plotting():
    X, Y = np.mgrid[-2:4.5:20j, -4:6:20j]
    Z = fun(np.array([X, Y]))

    fig = plt.figure()

    xs = polack_ribier_ethod(list([[1, 1], ]), 20)

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_wireframe(X, Y, Z)

    ax1 = fig.add_subplot(1, 2, 2)

    X, Y = np.mgrid[-1.5:1.1:30j, -1.25:6.:30j]
    Z = fun(np.array([X, Y]))

    ax1.contour(X, Y, Z, levels=40)

    # for point in np.array(xs):
    #     ax.scatter(point[0], point[1], fun(point), color='red')

    #ax.scatter(np.array(xs).T[0], np.array(xs).T[1], np.array([map(fun, np.array(xs).T[0], np.array(xs).T[1])]) , c='r')

    ax.plot(np.array(xs).T[0], np.array(xs).T[1], np.array(
        list(map(fun, np.array(xs)))).T, "x-", color='red')

    ax1.plot(np.array(xs).T[0], np.array(xs).T[1], "x-")

    print(f"x = {xs[-1][0]}, y = {xs[-1][1]}, z = {fun(xs[-1])}")

    plt.show()


plotting()
