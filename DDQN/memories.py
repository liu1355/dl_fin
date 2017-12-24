# Memory and state management for batch processing, model saving and evaluation

import random
import numpy as np
import os
from DDQN.abc import Memory
from DDQN.utils import save_npy, load_npy

class VectorMemory(Memory):
    """Function 1: init
       Function 2: add
       Function 3: Select memory batch, split into categorical sub-batches, process action batch into positional vector"""

    def __init__(self, memory_size, state_size, action_size):
        super(VectorMemory, self).__init__(memory_size)
        self.memory = [None] * memory_size # Empty state
        self.state_size = state_size
        self.action_size = action_size

    def add(self, state, action, reward, next_state, terminal):
        super(VectorMemory, self).add(state, action, reward, next_state, terminal)
        self.memory[self.i] = (state, action, reward, next_state, terminal)

    def sample(self, size):

        batch = np.array(random.sample(self.memory, size))
        state_batch = np.concatenate(batch[:,0])\
            .reshape(size,self.state_size)
        action_batch = np.concatenate(batch[:,1])\
            .reshape(size,self.action_size)
        reward_batch = batch[:,2]
        next_state_batch = np.concatenate(batch[:,3])\
            .reshape(size,self.state_size)
        done_batch = batch[:,4]
        # action processing
        action_batch = np.where(action_batch == 1)
        return state_batch,action_batch,reward_batch,next_state_batch,done_batch

class ReplayMemory(Memory):
    """Source: tambetm"""

    def __init__(self, config, model_dir): # REMINDER: model_dir DOES NOT EXIST AS SUMMARY WRITER NOT INCLUDED
        super(ReplayMemory, self).__init__(config.memory_size)
        self.model_dir = model_dir
        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.float32)
        self.states = np.empty((self.memory_size, config.input_size), dtype = np.int32)
        self.terminals = np.empty(self.memory_size, dtype = np.bool)
        self.history_length = config.history_length
        self.dims = config.input_size
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0

        # Pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length, self.dims), dtype = np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length, self.dims), dtype = np.float16)

    def add(self, state, action, reward, next_state, terminal):

        # Assert state.shape == self.dims
        assert len(state) == self.dims

        # Note: State is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.states[self.current, ...] = state
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        assert self.count > 0, "replay memory is empty, use at least --random_steps 1"

        # Normalize index to expected range, allows negative indexes
        index = index % self.count

        # If is not in the beginning of matrix
        if index >= self.history_length - 1:
            return self.states[(index - (self.history_length - 1)):(index + 1), ...]

        # Otherwise normalize indexes and use slower list based access
        else:
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.states[indexes, ...]

    def sample(self, size=None):
        if size is None:
            size = self.batch_size

        # Memory must include post-state, pre-state and history
        assert self.count > self.history_length

        # Sample random indices
        indexes = []

        while len(indexes) < size:
            # Find random index
            while True:
                # Sample one index (ignore states wrapping over)
                index = random.randint(self.history_length, self.count - 1)
                # If wrap over current pointer, then get new one
                if index >= self.current and index - self.history_length < self.current:
                    continue
                # If wrap over episode end, then get new one
                # Note: Post-state (last screen) can be terminal state!
                if self.terminals[(index - self.history_length):index].any():
                    continue
                # Otherwise use this index
                break
            # Note: Having index first is the fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self.prestates, actions, rewards, self.poststates, terminals

    def save(self):
        for idx, (name, array) in enumerate(
                zip(['actions', 'rewards', 'states', 'terminals', 'prestates', 'poststates'],
                    [self.actions, self.rewards, self.states, self.terminals, self.prestates, self.poststates])):
            save_npy(array, os.path.join(self.model_dir, name)) # REMINDER: model_dir DOES NOT EXIST AS SUMMARY WRITER NOT INCLUDED

    def load(self):
        for idx, (name, array) in enumerate(
                zip(['actions', 'rewards', 'states', 'terminals', 'prestates', 'poststates'],
                    [self.actions, self.rewards, self.states, self.terminals, self.prestates, self.poststates])):
            array = load_npy(os.path.join(self.model_dir, name)) # REMINDER: model_dir DOES NOT EXIST AS SUMMARY WRITER NOT INCLUDED