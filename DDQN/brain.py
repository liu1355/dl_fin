# General DDQN brain architecture

import numpy as np
import tensorflow as tf

from DDQN.abc import Brain
from DDQN.abc import ModelSaver


class TFBrain(Brain, ModelSaver):
    """Function 1: Model saver loadable from disk, prediction and target networks, optimizer
       Function 2: Network topology
       Function 3a & 3b: Update target Q net
       Function 4: Bellman equation
       Function 5: Training
       Function 6: Prediction
       Function 7: Get learning rate"""

    def __init__(self, config, sess, load, step_op):
        super(TFBrain, self).__init__(config, sess)

        # Network variables set to blank for createNet
        # Note: The dueling network has separated the state-dependent action values and state values.
        # Note: The final convolution layer is flattened to less costly abstract layer of lower dimension.

        self.w, self.s_t, self.s_t_flat, self.l1, self.value_hid, self.adv_hid = {}, {}, {}, {}, {}, {}
        self.value, self.advantage, self.q, self.l2 = {}, {}, {}, {}

        self.sess = sess

        # Training network
        with tf.variable_scope('prediction'):
            self._createNet('p') # Abstract attribute reference

            # Argmax of Q values denote best action
            self.q_action = tf.argmax(self.q['p'], dimension = 1)

            # REMINDER: TENSORBOARD SUMMARY WRITER HERE

        # Target network
        # Note: Target network of a dueling net is separate net; parameters defined in game environment and here.
        with tf.variable_scope('target'):
            self._createNet('target')
            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.q['target'], self.target_q_idx)

        # Copy the corresponding prediction values to the target Q values.
        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w['p'].keys():
                self.t_w_input[name] = tf.placeholder('float32', self.w['target'][name].get_shape().as_list(),
                                                      name=name)
                self.t_w_assign_op[name] = self.w['target'][name].assign(self.t_w_input[name])

        # Optimization
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name = 'target_q_t')
            self.action = tf.placeholder('int64', [None], name = 'action')

            # Note: Use Huber loss function, which is less sensitive to outliers; see static method below.
            # Note: Contains abstract attribute references.
            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name = 'action_one_hot')
            q_acted = tf.reduce_sum(self.q['p'] * action_one_hot, reduction_indices=1, name = 'q_acted')

            delta = self.target_q_t - q_acted
            self.global_step = tf.Variable(0, trainable = False)

            self.loss = tf.reduce_mean(TFBrain._clipped_error(delta), name='loss')

            # Learning rate exponential decay
            self.learning_rate_step = tf.placeholder('int64', None, name = 'learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_params['min'],
                                               tf.train.exponential_decay(
                                                   self.learning_rate_params['lr'],
                                                   self.learning_rate_step,
                                                   self.learning_rate_params['decay_step'],
                                                   self.learning_rate_params['decay'],
                                                   staircase = True))

            # Standard RMSprop optimizer (adjustable parameters)
            # Note: Motivation is the magnitude (root mean squared) of previous gradients divided by current gradients.
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op,
                                                   momentum = 0.95, epsilon = 0.01).minimize(self.loss)

            # REMINDER: TENSORBOARD SUMMARY WRITER HERE
            # Tags: Average reward, average loss, average Q, episode max reward, episode min reward, episode average reward
            # Tags: Episode no. of games, training learning rate etc.
            # Note: Remember to add inject summary function for the above tags.

        # Initialize the variables, create saver, and load model if load is true.
        tf.global_variables_initializer().run()
        self._saver = tf.train.Saver(self.w['p'].values() + [step_op], max_to_keep = 30)

        if load:
            self.load_model()

    def _createNet(self, name):

        # Network topology, (Wang, Shaul et al, 2016)
        # Note: Dueling and non-dueling toggleable in game environment.

        if name not in self.w:
            self.w[name] = {}
        activation_fn = tf.nn.relu
        self.s_t[name] = tf.placeholder('int32',
                                        [None, self.history_length, self.input_size], name='s_t.{0}'.format(name))
        self.s_t_flat[name] = tf.reshape(self.s_t[name], [-1, self.input_size * self.history_length])
        self.l1[name], self.w[name]['l1_w'], self.w[name]['l1_b'] = self.nn_layer(self.s_t_flat[name], self.num_hidden,
                                                                                  'l1.{0}'.format(name), activation_fn)

        if self.dueling:
            self.value_hid[name], self.w[name]['l2_val_w'], self.w[name]['l2_val_b'] = \
                self.nn_layer(self.l1[name], self.num_hidden, 'value_hid.{0}'.format(name), activation_fn)

            self.adv_hid[name], self.w[name]['l2_adv_w'], self.w[name]['l2_adv_b'] = \
                self.nn_layer(self.l1[name], self.num_hidden, 'adv_hid.{0}'.format(name), activation_fn)

            self.value[name], self.w[name]['val_w_out'], self.w[name]['val_w_b'] = \
                self.nn_layer(self.value_hid[name], 1, 'value_out.{0}'.format(name))

            self.advantage[name], self.w[name]['adv_w_out'], self.w[name]['adv_w_b'] = \
                self.nn_layer(self.adv_hid[name], self.action_size, 'adv_out.{0}'.format(name))

            # Average dueling

            self.q[name] = self.value[name] + (self.advantage[name]
                                               - tf.reduce_mean(self.advantage[name], reduction_indices=1,
                                                                keep_dims = True))

        # Regular deep Q net
        else:
            self.l2[name], self.w[name]['l2_w'], self.w[name]['l2_b'] = self.nn_layer(self.l1[name], self.num_hidden,
                                                                                      'l2.{0}'.format(name),
                                                                                      activation_fn)
            self.q[name], self.w[name]['q_w'], self.w[name]['q_b'] = self.nn_layer(self.l2[name], self.action_size,
                                                                                   name='q.{0}'.format(name))

    def init(self):
        self.update_target_q_network()

    def update_target_q_network(self):
        for name in self.w['p'].keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w['p'][name].eval()})

    # Bellman equation
    def calc_target(self, s_t_plus_1, terminal, reward, discount, double_q):

        if double_q:
            pred_action = self.q_action.eval({self.s_t['p']: s_t_plus_1})

            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
                self.s_t['target']: s_t_plus_1,
                self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
            })
            target_q_t = (1. - terminal) * discount * q_t_plus_1_with_pred_action + reward

        else:
            q_t_plus_1 = self.q['target'].eval({self.s_t['target']: s_t_plus_1})

            terminal = np.array(terminal) + 0.
            max_q_t_plus_1 = np.max(q_t_plus_1, axis = 1)
            target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

        return target_q_t

    def train(self, s_t, target_q_t, action, step):
        _, q_t, loss = self.sess.run([self.optim, self.q['p'], self.loss], {
                self.target_q_t: target_q_t,
                self.action: action,
                self.s_t['p']: s_t,
                self.learning_rate_step: step,
            })

    def predict(self, s_t, getVal = False):

        if not getVal:
            action = self.q_action.eval({self.s_t['p']: [s_t]})[0]
            return action

        # Reiterate to accumulate Q if not normal prediction
        else:
            vals = np.empty(shape=(s_t.shape[0], self.action_size))
            for i, item in enumerate(s_t):
                val = self.sess.run([self.q['p']], {self.s_t['p']: [[item]]})
                vals[i, :] = val[0][0]
            return vals

    def getLR(self, step):
        return self.learning_rate_op.eval({self.learning_rate_step: step})

    @staticmethod
    def _weight_init():
        return tf.truncated_normal_initializer(0, 0.2)

    @staticmethod
    def _bias_init():
        return tf.constant_initializer(0.01)

    # For use in createNet.
    @staticmethod
    def nn_layer(input_tensor, output_dim, name, act = None):
        shape = input_tensor.get_shape().as_list()
        with tf.variable_scope(name):
            w = tf.get_variable('Matrix', [shape[1], output_dim], tf.float32, initializer=TFBrain._weight_init())
            b = tf.get_variable('bias', [output_dim], initializer=TFBrain._bias_init())

            out = tf.nn.bias_add(tf.matmul(tf.to_float(input_tensor), w), b)

            if act != None:
                return act(out), w, b
            else:
                return out, w, b

    # Huber loss
    @staticmethod
    def _clipped_error(x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)