""" Utility classes and functions. """
from collections import namedtuple
import numpy as np

from kilobot.shapes.shapes import *


Packet = namedtuple('Packet', [
    'id',
    'gradient',
    'position',
    'internal_position',
    'orientation',
    'stationary',
    'following',
])


scale = 6                    # scale factor
dt = 0.05 * scale            # update time step
startup_time = 40            # fixed startup time
follow_distance = 0.05       # edge follow distance
gradient_distance = 0.06     # distance for gradient formation
hop_count_threshold = 4      # hysteresis threshold
yield_distance = 0.07        # emergency stop distance
speed = 0.02 / scale         # base speed
speed_std = 0.2              # standard deviation for speed variance
turn_rate = 3.0 / scale      # turning rate in rad/s
noise_std = 0.05             # standard deviation for noisy distance measurement
communication_radius = 0.15  # communication radius between nodes
loss_rate = 0.05             # rate of communication failures


shape = Rectangle(np.array([0.10, 0.15]))  # kilobot swarm shape
swarm_size = 4                             # side length of triangle swarm


animate_dt = 100        # animation update time in ms
figsize = (6.40, 4.80)  # figure size
xlim = (-0.20, 0.40)    # x-axis limits
ylim = (-0.40, 0.30)    # y-axis limits
marker_size = 130.0     # kilobot size