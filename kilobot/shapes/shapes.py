""" Shape objects. """
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Shape(ABC):
    @abstractmethod
    def is_inside(self, position: np.ndarray) -> bool:
        """ If the position is inside the shape. """
        pass

    @abstractmethod
    def plot(self) -> None:
        """ Plot shape boundary. """
        pass

    @abstractmethod
    def reset(self) -> None:
        """ Reset plot. """
        pass


class TestShape(Shape):
    """ Test empty shape. """
    def __init__(self) -> None:
        """ Test shape initializer. """
        super().__init__()
    
    def is_inside(self, position: np.ndarray) -> bool:
        """ If the position is inside the shape. """
        return False
    
    def plot(self) -> None:
        """ Plot shape boundary. """
        return
    
    def reset(self) -> None:
        """ Reset plot. """
        return


class Rectangle(Shape):
    """ Rectangle shape. """
    def __init__(self, corner: np.ndarray = np.array([0.25, 0.50])) -> None:
        """
        Rectangle initializer.
        
        Assumes bottom left corner is at (0, 0) and sides are axis-aligned.

        Args:
            corner (np.ndarray): Top right corner coordinate.
        """
        if corner.shape != (2,):
            raise ValueError('invalid corner coordinate')
        self._x_max, self._y_max = corner
        self._boundary = np.array([
            [        0.0,         0.0],
            [self._x_max,         0.0],
            [self._x_max, self._y_max],
            [        0.0, self._y_max],
            [        0.0,         0.0]
        ])
        self._boundary_plot = None

    def is_inside(self, position: np.ndarray) -> bool:
        """ If the position is inside the shape. """
        if position.shape != (2,):
            raise ValueError('invalid position')
        x, y = position
        return x >= 0 and x <= self._x_max and y >= 0 and y <= self._y_max
    
    def plot(self) -> None:
        """ Plot shape boundary. """
        if self._boundary_plot is None:
            self._boundary_plot, = plt.plot([], [], '--k')
        self._boundary_plot.set_data(self._boundary[:, 0], self._boundary[:, 1])

    def reset(self) -> None:
        """ Reset plot. """
        self._boundary_plot = None


class Donut(Shape):
    """ Rectangle with hole shape. """
    def __init__(self, corner: np.ndarray = np.array([0.13, 0.13])) -> None:
        """
        Donut initializer.
        
        Assumes bottom left corner is at (0, 0) and sides are axis-aligned. Thickness is 0.03.

        Args:
            corner (np.ndarray): Top right corner coordinate.
        """
        super().__init__()

        if corner.shape != (2,):
            raise ValueError('invalid corner coordinate')
        self._x_max, self._y_max = corner
        if self._x_max < 0.1 or self._y_max < 0.1:
            raise ValueError('shape is too small')
        self._outer_boundary = np.array([
            [        0.0,         0.0],
            [self._x_max,         0.0],
            [self._x_max, self._y_max],
            [        0.0, self._y_max],
            [        0.0,         0.0]
        ])
        self._inner_boundary = np.array([
            [              0.03,               0.03],
            [self._x_max - 0.03,               0.03],
            [self._x_max - 0.03, self._y_max - 0.03],
            [              0.03, self._y_max - 0.03],
            [              0.03,               0.03]
        ])
        self._outer_boundary_plot = None
        self._inner_boundary_plot = None

    def is_inside(self, position: np.ndarray) -> bool:
        """ If the position is inside the shape. """
        if position.shape != (2,):
            raise ValueError('invalid position')
        x, y = position
        if x < 0 or x > self._x_max or y < 0 or y > self._y_max:
            return False
        return not (x >= 0.03 and x <= self._x_max - 0.03 and y >= 0.03 and y <= self._y_max - 0.03)
    
    def plot(self) -> None:
        """ Plot shape boundary. """
        if self._outer_boundary_plot is None:
            self._outer_boundary_plot, = plt.plot([], [], '--k')
        if self._inner_boundary_plot is None:
            self._inner_boundary_plot, = plt.plot([], [], '--k')
        self._outer_boundary_plot.set_data(self._outer_boundary[:, 0], self._outer_boundary[:, 1])
        self._inner_boundary_plot.set_data(self._inner_boundary[:, 0], self._inner_boundary[:, 1])

    def reset(self) -> None:
        """ Reset plot. """
        self._outer_boundary_plot = None
        self._inner_boundary_plot = None


class L(Shape):
    """ L shape. """
    def __init__(self) -> None:
        """ L initializer. """
        super().__init__()
        self._boundary = np.array([
            [0.00, 0.00],
            [0.15, 0.00],
            [0.15, 0.07],
            [0.07, 0.07],
            [0.07, 0.15],
            [0.00, 0.15],
            [0.00, 0.00]
        ])
        self._boundary_plot = None

    def is_inside(self, position: np.ndarray) -> bool:
        """ If the position is inside the shape. """
        if position.shape != (2,):
            raise ValueError('invalid position')
        x, y = position
        return (
            (x >= 0 and x <= 0.07 and y >= 0 and y <= 0.15) or
            (x >= 0 and x <= 0.15 and y >= 0 and y <= 0.07)
        )
    
    def plot(self) -> None:
        """ Plot shape boundary. """
        if self._boundary_plot is None:
            self._boundary_plot, = plt.plot([], [], '--k')
        self._boundary_plot.set_data(self._boundary[:, 0], self._boundary[:, 1])

    def reset(self) -> None:
        """ Reset plot. """
        self._boundary_plot = None

