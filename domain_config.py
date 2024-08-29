"""
@author: WenXuan Yuan
Email: wenxuan.yuan@qq.com
"""

import numpy as np
import random
import scipy.io
from matplotlib import pyplot as plt


# domain parameters:
class D1:
    """Domain 1 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 2
        self.k_2 = -1
        self.d = 2

        self.u_positive = -2
        self.u_negative = -1

        self.t = 30

        self.x_boundary = [-380, 20]


class D2:
    """Domain 2 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 6
        self.k_2 = -3
        self.d = 15

        self.u_positive = -1
        self.u_star = 3
        self.u_negative = 2

        self.t = 10

        self.x_boundary = [-70, 180]


class D3:
    """Domain 3 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 6
        self.k_2 = -1
        self.d = 10

        self.u_positive = 3
        self.u_negative = 5

        self.t = 90

        self.x_boundary = [1260, 1800]


class D4:
    """Domain 4 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 6
        self.k_2 = -1
        self.d = 10

        self.u_positive = 4
        self.u_negative = 5

        self.t = 90

        self.x_boundary = [1260, 1720]


class D5:
    """Domain 5 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 12
        self.k_2 = -1
        self.d = -30

        self.u_positive = 11
        self.u_negative = 9

        self.t = 0.3

        self.x_boundary = [-11, 0.5]


class k2gt0_D5:
    """Domain 5 parameters"""

    def __init__(self):
        self.k2gt0 = True
        self.k_1 = 16
        self.k_2 = 2
        self.d = -6

        self.u_positive = 2
        self.u_negative = -3

        self.t = 0.5

        self.x_boundary = [-18.5, 17.5]


class D6:
    """Domain 6 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 2
        self.k_2 = -1 / 2
        self.d = -3 / 2

        self.u_positive = 4
        self.u_negative = 1

        self.t = 100

        self.x_boundary = [-340, 40]


class k2gt0_D6:
    """Domain 6 parameters"""

    def __init__(self):
        self.k2gt0 = True
        self.k_1 = 2
        self.k_2 = 1
        self.d = 3

        self.u_positive = 1
        self.u_negative = -2

        self.t = 5.0

        self.x_boundary = [-1.0, 31.0]


class D7:
    """Domain 7 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 3
        self.k_2 = -1
        self.d = 0

        self.u_positive = 3
        self.u_negative = -1

        self.t = 1.0

        self.x_boundary = [-5.0, 11.0]


class D8:
    """Domain 8 parameters"""

    def __init__(self):
        self.k2gt0 = False
        self.k_1 = 3
        self.k_2 = -1
        self.d = 0

        self.u_positive = 0
        self.u_negative = -1

        self.t = 1.0

        self.x_boundary = [-5.0, 1.0]


def step_function(x, u_positive, u_negative):
    """Step function"""
    if x >= 0:
        return u_positive
    elif x < 0:
        return u_negative