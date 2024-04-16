""" Main script. """
import numpy as np

from kilobot.node import Node
from kilobot.swarm import Swarm
from kilobot.utils import swarm_size


def generate_swarm() -> Swarm:
    """ Generate a swarm. """
    swarm = Swarm()

    # generate seeds
    gradient_seed = Node(0, np.array([-0.017, 0.0]), is_seed=True, is_gradient_seed=True)
    seeds = [
        Node(1, np.array([0.017,  0.00]), is_seed=True),
        Node(2, np.array([0.000,  0.03]), is_seed=True),
        Node(3, np.array([0.000, -0.03]), is_seed=True)
    ]
    swarm.add_node(gradient_seed)
    for seed in seeds:
        swarm.add_node(seed)
    
    # generate triangle cluster
    size = swarm_size
    positions = np.array([
        [0.050 * j, -0.07 - 0.045 * i] for i in range(size) for j in range(i + 1)
    ])
    for i, position in enumerate(positions):
        node = Node(i + 4, position)
        swarm.add_node(node)

    # add communication
    for i in range(4 + len(positions)):
        for j in range(i + 1, 4 + len(positions)):
            swarm.add_pipe(i, j)
            swarm.add_pipe(j, i)

    return swarm


if __name__ == '__main__':
    # generate a random graph with [30] nodes
    swarm = generate_swarm()
    
    print("========= Starting now =========")
    print("Close figure to end the simulation")
    swarm.run()              # start threads in nodes
    swarm.animation_setup()  # set up plotting
    print("Close figure to stop.....")
    swarm.stop()             # send stop signal
    print("========== Terminated ==========")

