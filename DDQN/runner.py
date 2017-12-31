import random
import argparse
import tensorflow as tf
import numpy as np
import pickle

from DDQN.agent import Agent
from DDQN.environment import Game
from DDQN.environment import get_config

# Set random seed
tf.set_random_seed(123)
random.seed(123)

class Runner(object):
    """Runner object for main run function"""

    pass

# Deploys DDQN algorithm.
def main(args):
    with tf.Session() as sess:

        config = get_config(args)
        saveloc = os.getcwd()

        w = Game(config, args.maxgamelen, args.random, saveloc)

        runner = Runner(w, config, saveloc, args.load, sess)

        if args.train:
            runner.train()
        else:
            runner.run()

class Stats(object):
    """Recording statistics while running"""

    pass