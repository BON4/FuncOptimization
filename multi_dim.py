import numpy as np
import matplotlib.pyplot as plt
import math
# Поллака-Рибьера
# Найскорейшого спуска


def fun(x: np.float, y: np.float) -> np.float:
    """
    Put your function here
    """
    return np.power((np.power(y, 2) + x - 11), 2)+np.power((x+np.power(y, 2)-7), 2)
