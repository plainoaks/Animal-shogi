import numpy as np
import copy
import tensorflow as tf
from tensorflow.python import keras as K
from tensorflow.keras import backend as B
import network

class REINFORCE:

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.weights = self.model.trainable_weights
        self.set_updater(self.model.optimizer)

    def set_updater(self, optimizer):
        actions = tf.compat.v1.placeholder(dtype='int32', shape=(None))
        rewards = tf.compat.v1.placeholder(dtype='float32', shape=(None))
        one_hot_actions = tf.one_hot(actions, 144, axis=1)
        probs = self.model.output
        selected_probs = tf.reduce_sum(one_hot_actions*probs, axis=1)
        clip_probs = tf.clip_by_value(selected_probs, 1e-10, 1.0)
        loss = - tf.math.log(clip_probs) * rewards
        loss = tf.reduce_mean(loss)
        updates = optimizer.get_updates(loss=loss, params=self.weights)
        self._updater = B.function(
            inputs=[self.model.input, actions, rewards],
            outputs=[loss],
            updates=updates
        )


    def update_policy(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        res = self._updater([states, actions, rewards])
        print(res)
        
