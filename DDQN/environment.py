# Game environment peroperly defining state signals, using Quantopian pipelines as a data source
# Portfolio factor models, factor risk exposure, technical signals and other econometric methods

import numpy as np
import pickle
import os

from pipeline import TVF, FFM, AlphaFactors, EconomicData

class DataSource(TVF, FFM, AlphaFactors, EconomicData):
    """Quantopian pipeline here"""

    def __init__(self, d_TVF, d_FFM, d_Alpha, d_Econ):
        super(self.__class__, self).__init__()
        self.d_TVF = d_TVF
        self.d_FFM = d_FFM
        self.d_Alpha = d_Alpha
        self.d_Econ = d_Econ
    # WIP: Insert dataframe manipulation and cleanup here.
    def env_data(self):
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

    env_name = 'Game'
    num_hidden = 5
    action_size = 3
    max_reward = 16.
    min_reward = -16.
    input_size = 2
    num_st = 5
    samepenalty = 1.5
    rewardscale = 1.0

class DQNConfig(AgentConfig, EnvironmentConfig):
    pass

def get_config(args):
    if args.model == 'Trading':
        config = DQNConfig
    else:
        raise ValueError("Bad model {0}".format(args.model))

    return config

class Game(object):
    """Defining states, rewards, portfolio positions here"""

    S_FLAT = 2
    S_LONG = 1
    S_SHORT = 0
    A_NOTHING = 2
    A_BUY = 1
    A_SELL = 0

    def __init__(self, config, maxlen):
        super(self.__class__, self).__init__()
        self.num_st = config.num_st
        self.same_penalty = config.samepenalty
        self.reward_scale = config.rewardscale

        self.name = 'Game'
        self.maxlen = maxlen

        self.state = np.empty(config.input_size)
        self.data = DataSource(TVF, FFM, AlphaFactors, EconomicData)
        self.qhandler = QHandler() # WIP: Fill later

        self._initStats() # WIP: Fill later
        self.reset() # WIP: Consider removing

    def _initStats(self):
        self.glen = []
        self.gpen = []
        self.grew = []
        self.costs = []
        self.rewards = []
        self.actionsmem = []
        self.numChange = 0

    def reset(self):
        pass # WIP: Consider removing

    def _fill_state(self):
        self.state[0] = self.data.get_val()
        self.state[1] = self.position

    def _getUpdate(self, action):
        if action == Game.A_NOTHING:
            return self.position, 0, 0, False
        v1 = self.data.get_val(1)
        v0 = self.data.get_val()
        if self.position == Game.S_FLAT:
            if action == Game.A_BUY:
                return Game.S_LONG, 0, self.reward_scale * (v1 - v0), False
            if action == Game.A_SELL:
                return Game.S_SHORT, 0, self.reward_scale * (v0 - v1), False
        if self.position == Game.S_SHORT:
            if action == Game.A_BUY:
                return Game.S_FLAT, 0, self.last[1] - v0, True
            if action == Game.A_SELL:
                return Game.S_SHORT, self.same_penalty, 0, False
        if self.position == Game.S_LONG:
            if action == Game.A_BUY:
                return Game.S_LONG, self.same_penalty, 0, False
            if action == Game.A_SELL:
                return Game.S_FLAT, 0, v0 - self.last[1],

    def step(self, action):
        if action == Game.A_NOTHING:
            return self.position, 0, 0, False
        v1 = self.data.get_val(1)
        v0 = self.data.get_val()
        if self.position == Game.S_FLAT:
            if action == Game.A_BUY:
                return Game.S_LONG, 0, self.reward_scale * (v1 - v0), False
            if action == Game.A_SELL:
                return Game.S_SHORT, 0, self.reward_scale * (v0 - v1), False
        if self.position == Game.S_SHORT:
            if action == Game.A_BUY:
                return Game.S_FLAT, 0, self.last[1] - v0, True
            if action == Game.A_SELL:
                return Game.S_SHORT, self.same_penalty, 0, False
        if self.position == Game.S_LONG:
            if action == Game.A_BUY:
                return Game.S_LONG, self.same_penalty, 0, False
            if action == Game.A_SELL:
                return Game.S_FLAT, 0, v0 - self.last[1], True

        penalty = 0.0
        reward = 0
        position = self.position
        done = False
        if action != Game.A_NOTHING:
            position, penalty, reward, done = self._getUpdate(action)
            self.cumpenalty += penalty
            if not self.last:
                self.last = (self.data.clock, self.data.get_val())
            if done:
                self.glen.append(self.data.clock - self.last[0])
                self.gpen.append(self.cumpenalty)
                self.grew.append(reward)
            self.actionsmem.append([self.data.clock, action])
            self.numChange += 1

        self.position = position
        self.data.clock += 1
        self._fill_state()

        if self.data.clock == self.maxlen:
            penalty += self.same_penalty * 50.0  # Very bad
            done = True

            return self._get_state(), (reward - penalty), done, None

    @staticmethod
    def translatePosition(position):
        if position == Game.S_FLAT:
            return 'FLAT'
        elif position == Game.S_LONG:
            return 'LONG'
        else:
            return 'SHORT'

    @staticmethod
    def translateAction(action):
        if action == Game.A_NOTHING:
            return 'NOTHING'
        elif action == Game.A_BUY:
            return 'BUY'
        elif action == Game.A_SELL:
            return 'SELL'

    def showQS(self, qs):
        self.qhandler.showQS(qs, Game.translateAction)


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
        entries = np.empty(shape=(mlen, self.input_size))
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
