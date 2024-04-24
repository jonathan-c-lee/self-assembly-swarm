""" Kilobot swarm. """
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from kilobot.node import Node
from kilobot.pipe import Pipe
from kilobot.utils import animate_dt, figsize, xlim, ylim, marker_size


plt.rcParams['figure.figsize'] = figsize
dirname = os.path.dirname(__file__)


class Swarm:
    """ Kilobot swarm. """
    def __init__(self) -> None:
        """ Swarm initializer. """
        self._nodes = []
        self._pipes = []

        self._dt = animate_dt
        self._fig = plt.figure()
        self._ax = plt.axes(xlim=xlim, ylim=ylim)
        self._ax.set_aspect('equal', 'box')
        self._plot = None
        self._cb = None
        self._animation = None

    def add_node(self, node: Node) -> None:
        """ Add node to swarm. """
        self._nodes.append(node)

    def add_pipe(self, writer: int, reader: int) -> None:
        """
        Add communication pipe to swarm.
        
        Args:
            writer (int): Pipe writer index.
            reader (int): Pipe reader index.
        """
        pipe = Pipe(self._nodes[writer], self._nodes[reader])
        self._nodes[writer].add_writer(pipe)
        self._nodes[reader].add_reader(pipe)
        self._pipes.append(pipe)

    def run(self) -> None:
        """ Run all nodes. """
        # start running threads
        for node in self._nodes:
            node.start()

    def stop(self) -> None:
        """ Stop all nodes. """
        # send stop signal
        for node in self._nodes:
            node.terminate()

        # calculate error
        positions = self.node_positions()
        locals = self.node_localizations()
        errors = np.linalg.norm(positions - locals, axis=1)
        print('Distance Errors:')
        print(errors)
        print('Average Distance Error:')
        print(np.average(errors[4:]))

        # plot error
        plt.figure()
        ax = plt.axes(xlim=xlim, ylim=ylim)
        ax.set_aspect('equal', 'box')
        ax.scatter(
            positions[:, 0], positions[:, 1], c='None', s=marker_size, edgecolors='b', label='true position'
        )
        ax.scatter(
            locals[:, 0], locals[:, 1], c='None', s=marker_size, edgecolors='r', label='localized position'
        )
        ax.legend(loc='lower right')
        for start, end in zip(positions, locals):
            x = [start[0], end[0]]
            y = [start[1], end[1]]
            ax.plot(x, y, 'black')
        Node.plot_shape()
        plt.savefig(os.path.join(dirname, '../plots/tmp_RENAME.png'), transparent=True, dpi=300, bbox_inches='tight')
        plt.show()

        for node in self._nodes:
            node.join()

    def node_positions(self) -> np.ndarray:
        """ Collect all node positions. """
        return np.array([node.position for node in self._nodes])
    
    def node_localizations(self) -> np.ndarray:
        return np.array([node.localization for node in self._nodes])
    
    def node_gradients(self) -> np.ndarray:
        """ Collect all node gradients. """
        return np.array([node.gradient for node in self._nodes])

    def animate(self, i):
        """ Animation method. """
        positions = self.node_positions()
        gradients = self.node_gradients()
        if self._plot is None:
            self._plot = self._ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=gradients,
                s=marker_size,
                edgecolors='black',
                vmin=0.0,
                vmax=float(len(self._nodes) / 2)
            )
            self._cb = self._fig.colorbar(self._plot, ax=self._ax)
        self._plot.set_offsets(positions)
        self._plot.set_array(gradients)
        Node.plot_shape()
        if i % 5 == 0:
            plt.savefig(os.path.join(dirname, f'../images/tmp_{i // 5}.png'), dpi=150, bbox_inches='tight')
        return self._plot,

    def animation_setup(self) -> None:
        """ Initialize animation. """
        self._animation = animation.FuncAnimation(
            self._fig, self.animate, interval=self._dt, blit=False, cache_frame_data=False
        )
        plt.show()

