# Game environment peroperly defining state signals, using Quantopian pipelines as a data source
# Portfolio factor models, factor risk exposure, technical signals and other econometric methods

import numpy as np
import pickle
import os

class DataSource(object):
    """Quantopian pipeline here"""
    pass

class Game(object):
    """Defining states, rewards, portfolio positions here"""
    pass


class AgentConfig(object):
    """Set Agent parameters here"""

    scale = 30

    max_step = 600 * scale
    memory_size = 10 * scale

    batch_size = 64
    discount = 0.99
    target_q_update_step = 1 * scale

    learning_rate_params = {'lr': 0.002, 'min': 0.0001, 'decay': 0.96, 'decay_step': 10 * scale}

    ep_end = 0.00
    ep_start = 1.
    ep_end_t = int(max_step * 0.8)

    history_length = 1
    train_frequency = 1
    learn_start = 0

    double_q = True
    dueling = True

    randact = [0.2, 0.2, 0.6]

    test_step = 20 * scale

class EnvironmentConfig(object):
    """Set Game parameters here"""

    env_name = 'TradingGame'
    num_hidden = 5
    action_size = 3
    max_reward = 16.
    min_reward = -16.
    input_size = 2
    num_st = 5
    amp_scale = 2
    samepenalty = 1.5
    rewardscale = 1.0

class DQNConfig(AgentConfig, EnvironmentConfig):
    pass

def get_config(args):
    if args.model == 'trading':
        config = DQNConfig
    else:
        raise ValueError("Bad model {0}".format(args.model))

    return config

# WIP: Look into this with Tensorboard writer.
class QHandler(object):
    """For visualising Q value graphs."""

    def __init__(self, init_data, num_st, input_size, output_size):
        self.num_st = num_st
        self._qs = []
        self.input_size = input_size
        self.output_size = output_size
        self._prepQS(init_data)

    def _prepQS(self, init_data):
        mlen = self.output_size * self.num_st
        entries = np.empty(shape = (mlen, self.input_size))
        pos = 0
        for j in range(self.output_size):
            for x in range(self.num_st):
                entries[pos] = [init_data[x], j]
                pos += 1
        self.testSet = entries

    def sampleQS(self, predict_fn):
        qv = predict_fn(self.testSet, True)
        self._qs.append(qv)
        return qv

    def getQS(self):
        return self._qs

    @staticmethod
    def _showQSDetail(ds, translator):
        for i in range(ds.shape[0]):
            print('{0} : {1}'.format(str(ds[i,]), translator(np.argmax(ds[i,]))))

    def showQS(self, qs, translator):
        print('for short:')
        QHandler._showQSDetail(qs[0:self.num_st], translator)
        print('for long:')
        QHandler._showQSDetail(qs[self.num_st:self.num_st * 2], translator)
        print('for flat:')
        QHandler._showQSDetail(qs[self.num_st * 2:self.num_st * 3], translator)