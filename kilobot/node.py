""" Kilobot node. """
from enum import Enum
from threading import Thread
import time
import numpy as np

from kilobot.pipe import Pipe
from kilobot.utils import (
    Packet,
    dt,
    startup_time,
    follow_distance,
    gradient_distance,
    hop_count_threshold,
    yield_distance,
    speed,
    turn_rate,
    speed_std,
    noise_std,
    loss_rate,
    shape
)


class Motion(Enum):
    """ Kilobot motion. """
    STOP = 0
    FORWARD = 1
    RIGHT = 2
    LEFT = 3


class State(Enum):
    """ Kilobot state. """
    START = 0
    WAIT = 1
    MOVE_OUTSIDE = 2
    MOVE_INSIDE = 3
    JOINED = 4


class Node(Thread):
    """ Kilobot agent node. """
    _dt = dt
    _startup_time = startup_time
    _follow_distance = follow_distance
    _gradient_distance = gradient_distance
    _shape = shape

    def __init__(
            self,
            uid: int,
            position: np.ndarray = np.array([0.0, 0.0]),
            is_seed: bool = False,
            is_gradient_seed: bool = False) -> None:
        """
        Node initializer.

        Args:
            uid (int): Unique ID.
            position (np.ndarray): Starting position.
            is_seed (bool): If kilobot is a seed. Defaults to False.
            is_gradient_seed (bool): If kilobot is the gradient seed. Defaults to False.
        """
        super().__init__()

        if position.shape != (2,):
            raise ValueError('invalid position')
        
        self._uid = uid
        self._position = np.copy(position)
        self._internal_position = np.copy(position)
        self._is_seed = is_seed
        self._is_gradient_seed = is_gradient_seed

        self._done = False   # termination flag
        self._outgoing = []  # list of pipes to write to
        self._incoming = []  # list of pipes to read from
        self._neighbors = {
            'distance': [],
            'gradient': []
        }

        self._orientation = None
        self._speed = np.random.normal(1.0, scale=speed_std) * speed
        self._yield_distance = yield_distance
        self._motion = Motion.STOP
        self._state = State.START
        self._time = 0
        self._local_id = -1
        self._previous_distance = np.inf
        self._gradient = 0
        self._hop_count = 0    # gradient hysteresis counter
        self._move_count = 0   # movement start counter
        self._shape_count = 0  # shape hysteresis counter

    @staticmethod
    def noisy_reading(reading: float) -> float:
        """ Generate a noisy distance reading. """
        return np.random.normal(1.0, scale=noise_std) * reading

    @classmethod
    def plot_shape(cls) -> None:
        """ Plot shape boundary. """
        cls._shape.reset()
        cls._shape.plot()
    
    @property
    def following(self) -> bool:
        """ Edge following state. """
        return self._state is State.MOVE_OUTSIDE or self._state is State.MOVE_INSIDE
    
    @property
    def stationary(self) -> bool:
        """ Stationary state. """
        return not self.following

    @property
    def position(self) -> np.ndarray:
        """ Position. """
        return self._position
    
    @property
    def localization(self) -> np.ndarray:
        """ Localized position. """
        return self._internal_position
    
    @property
    def gradient(self):
        """ Gradient. """
        return self._gradient
    
    @property
    def speed(self):
        """ Speed. """
        return self._speed
    
    @speed.setter
    def speed(self, speed: float):
        """ Set speed. """
        self._speed = speed

    def add_writer(self, outgoing: Pipe) -> None:
        """ Add pipe for outgoing packets. """
        self._outgoing.append(outgoing)

    def add_reader(self, incoming: Pipe) -> None:
        """ Add pipe for incoming packets. """
        self._incoming.append(incoming)

    def terminate(self) -> None:
        """ Terminate node. """
        self._done = True

    def send(self) -> None:
        """ Send packets. """
        data = Packet(
            self._local_id,
            self._gradient,
            self._position,
            self._internal_position,
            self._orientation,
            self.stationary,
            self.following
        )
        for pipe in self._outgoing:
            pipe.write(data)

    def update_neighbors(self) -> None:
        """ Update neighbor lists. """
        distance_neighbors = []
        gradient_neighbors = []
        for neighbor in self._incoming:
            data = neighbor.read()
            if data is None:
                continue
            # simulate lossy communication
            if np.random.random_sample() < loss_rate:
                continue
            distance_neighbors.append(data)
            if neighbor.distance < self._gradient_distance:
                gradient_neighbors.append(data)
        if len(distance_neighbors) != 0:
            self._neighbors['distance'] = distance_neighbors
        if len(gradient_neighbors) != 0:
            self._neighbors['gradient'] = gradient_neighbors

    def update_id(self) -> None:
        """ Update locally unique ID. """
        if self._local_id == -1:
            self._local_id = np.random.randint(100)
        else:
            for neighbor in self._neighbors['distance']:
                if self._local_id == neighbor.id:
                    self._local_id = -1
                    break

    def update_gradient(self) -> None:
        """ Perform gradient formation algorithm. """
        if self._is_gradient_seed:
            self._gradient = 0
        else:
            if len(self._neighbors['gradient']) == 0:
                return
            previous_gradient = self._gradient
            self._gradient = min(
                np.inf, min(self._neighbors['gradient'], key = lambda neighbor: neighbor.gradient).gradient
            )
            if self._gradient == np.inf:
                self._gradient = previous_gradient
            new_gradient = self._gradient + 1
            # gradient hop hysteresis
            if new_gradient > previous_gradient:
                if self._hop_count >= hop_count_threshold:
                    self._gradient = new_gradient
                    self._hop_count = 0
                else:
                    self._hop_count += 1
            else:
                self._gradient = new_gradient
                self._hop_count = 0

    def localize(self) -> None:
        """ Perform localization using greedy search to minimize trilateration equation. """
        # find all stationary neighbors
        neighbors = [neighbor for neighbor in self._neighbors['distance'] if neighbor.stationary]
        
        # update position
        if len(neighbors) < 3:
            return
        for neighbor in neighbors:
            vector = self._internal_position - neighbor.internal_position
            distance = np.linalg.norm(vector)
            vector /= distance
            measured_distance = self.noisy_reading(np.linalg.norm(self._position - neighbor.position))
            new_position = neighbor.internal_position + measured_distance * vector
            self._internal_position -= (self._internal_position - new_position) / 4

    def edge_follow(self) -> None:
        """ Edge follow nearest neighbor at desired distance. """
        # check for neighbors
        if len(self._neighbors['distance']) == 0:
            self._motion = np.random.choice(list(Motion))
            return

        # find distance to nearest neighbor
        distance = min([
            self.noisy_reading(np.linalg.norm(self._position - neighbor.position))
            for neighbor in self._neighbors['distance']
        ])

        # update movement
        if distance < self._follow_distance:
            self._motion = Motion.FORWARD if self._previous_distance < distance else Motion.LEFT
        else:
            self._motion = Motion.FORWARD if self._previous_distance > distance else Motion.RIGHT
        self._previous_distance = distance

    def yield_check(self) -> bool:
        """ Whether to yield to front edge follower. """
        for neighbor in self._neighbors['distance']:
            if not neighbor.following:
                continue
            # neighbor orientation may not exist yet
            if neighbor.orientation is None:
                continue
            vector = neighbor.position - self._position
            if self.noisy_reading(np.linalg.norm(vector)) > self._yield_distance:
                continue

            # check if edge follower is in front
            if np.dot([np.cos(neighbor.orientation), np.sin(neighbor.orientation)], vector) > 0.0:
                self._yield_distance = 1.75 * yield_distance
                return True
        self._yield_distance = yield_distance
        return False
    
    def initialize_orientation(self) -> None:
        """ Initialize node orientation. """
        nearest_neighbor = min(
            self._neighbors['distance'],
            key = lambda neighbor: np.linalg.norm(self._position - neighbor.position)
        )
        vector = self._position - nearest_neighbor.position
        vector /= np.linalg.norm(vector)
        angle = np.arccos(np.dot(np.array([1.0, 0.0]), vector))
        if vector[1] > 0.0:
            self._orientation = angle
        else:
            self._orientation = 2 * np.pi - angle
        self._orientation -= np.pi / 4  # bias towards clockwise direction

    def start_behavior(self) -> None:
        """ Start state behavior. """
        if self._is_seed:
            self._state = State.JOINED
            self._internal_position = np.copy(self._position)
        else:
            self._time += 1
            if self._time > self._startup_time:
                self._state = State.WAIT

    def wait_behavior(self) -> None:
        """ Wait state behavior. """
        if np.all([neighbor.stationary for neighbor in self._neighbors['distance']]):
            max_gradient = max([0] + [neighbor.gradient for neighbor in self._neighbors['gradient']])
            if self._gradient > max_gradient:
                if self._move_count > hop_count_threshold:
                    self._state = State.MOVE_OUTSIDE
                else:
                    self._move_count += 1
            elif self._gradient == max_gradient:
                for neighbor in self._neighbors['gradient']:
                    if neighbor.gradient != max_gradient:
                        continue
                    if self._local_id < neighbor.id:
                        self._move_count = 0
                        return
                # id count hysteresis
                if self._move_count > hop_count_threshold:
                    self._state = State.MOVE_OUTSIDE
                    self._move_count = 0
                else:
                    self._move_count += 1
            else:
                self._move_count = 0
        else:
            self._move_count = 0

    def outside_behavior(self) -> None:
        """ Moving outside shape behavior. """
        if self._orientation is None:
            self.initialize_orientation()
        if self.yield_check():
            self._motion = Motion.STOP
            return

        if self._shape.is_inside(self._internal_position):
            if self._shape_count > hop_count_threshold:
                self._state = State.MOVE_INSIDE
                self._shape_count = 0
            else:
                self._shape_count += 1
        else:
            self.edge_follow()
            self._shape_count = 0
    
    def inside_behavior(self) -> None:
        if self.yield_check():
            self._motion = Motion.STOP
            return

        if not self._shape.is_inside(self._internal_position):
            self._state = State.JOINED
            return

        gradient = min(
            self._neighbors['distance'],
            key = lambda neighbor: self.noisy_reading(np.linalg.norm(self._position - neighbor.position))
        ).gradient
        if self._gradient <= gradient:
            if self._shape_count > hop_count_threshold:
                self._state = State.JOINED
                self._shape_count = 0
            else:
                self._shape_count += 1
        else:
            self.edge_follow()
            self._shape_count = 0

    def update(self) -> None:
        """ Update node following self-assembly algorithm. """
        self.update_neighbors()
        self.update_id()
        self.update_gradient()
        self.localize()

        if self._state is State.START:
            self.start_behavior()
        elif self._state is State.WAIT:
            self.wait_behavior()
        elif self._state is State.MOVE_OUTSIDE:
            self.outside_behavior()
        elif self._state is State.MOVE_INSIDE:
            self.inside_behavior()
        elif self._state is State.JOINED:
            self._motion = Motion.STOP

    def move(self) -> None:
        """ Movement-based position update. """
        if self._motion is Motion.STOP or self._state is State.JOINED:
            return
        if self._motion is Motion.LEFT:
            self._orientation += self._dt * turn_rate
        elif self._motion is Motion.RIGHT:
            self._orientation -= self._dt * turn_rate
        self._position[0] += self._dt * self._speed * np.cos(self._orientation)
        self._position[1] += self._dt * self._speed * np.sin(self._orientation)

    def run(self) -> None:
        """ Run node. """
        while not self._done:
            start = time.time()

            self.send()
            if self._state is State.JOINED:
                self._motion = Motion.STOP
            else:
                self.update()
                self.move()
            
            end = time.time()
            time.sleep(max(self._dt - (end-start), 0))

