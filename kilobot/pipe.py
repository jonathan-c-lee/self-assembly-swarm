""" Pipe communication. """
from typing import Optional
import numpy as np

from kilobot.utils import Packet, communication_radius


class Pipe:
    """ Pipe for one-way communication between nodes. """
    _communication_radius = communication_radius

    def __init__(self, writer, reader) -> None:
        """
        Pipe initializer.

        Args:
            writer (Node): Writer node.
            reader (Node): Reader node.
        """
        self._writer = writer
        self._reader = reader

        self._data = None

    @property
    def distance(self) -> float:
        """ Communication distance. """
        return np.linalg.norm(self._reader.position - self._writer.position)

    def read(self) -> Optional[Packet]:
        """ Read method. """
        if self.distance < self._communication_radius:
            return self._data
        return None
    
    def write(self, data: Packet) -> None:
        """ Write method. """
        if self.distance < self._communication_radius:
            self._data = data

