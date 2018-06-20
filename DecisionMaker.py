import random
from random import randrange
import numpy as np

from Action import Action, ActionType
from MyEstimator import MyEstimator
import tensorflow as tf
import os


class DecisionMaker(MyEstimator):
    def __init__(self, screen_width, screen_height):
        self.conv_layers_kernel_size = [8, 4]
        self.conv_layers_stride_size = [4, 2]
        self.conv_layers_filters = [16, 32]
        self.dense_units = [256]
        output_dir = "decision_maker"
        for i in range(len(self.conv_layers_filters)):
            output_dir += "_conv_{}_{}".format(self.conv_layers_kernel_size[i], self.conv_layers_filters[i])
        for units in self.dense_units:
            output_dir += "_dense{}".format(units)
        os.makedirs(output_dir, exist_ok=True)
        model_dir = os.path.join(output_dir, "model.ckpt")
        super().__init__(model_dir=model_dir)
        self.screen_width = screen_width
        self.screen_height = screen_height

    def define_model(self, training_phase):
        features = dict()
        labels = dict()
        features['screen'] = tf.placeholder(dtype=tf.float32, shape=[None, self.screen_height, self.screen_width, 1])
        features['speed'] = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        labels['q_value'] = tf.placeholder(dtype=tf.float32, shape=[None, len(ActionType)])
        labels['action'] = tf.placeholder(dtype=tf.int32, shape=[None])
        # net = tf.cast(features['screen'], dtype=tf.float32)
        net = features['screen']
        for i in range(len(self.conv_layers_kernel_size)):
            net = tf.layers.conv2d(inputs=net,
                                   kernel_size=self.conv_layers_kernel_size[i],
                                   strides=self.conv_layers_stride_size[i],
                                   filters=self.conv_layers_filters[i],
                                   padding='same',
                                   activation=tf.nn.relu)
        net = tf.layers.flatten(inputs=net)
        net = tf.concat([net, features['speed']], axis=1)
        for units in self.dense_units:
            net = tf.layers.dense(inputs=net, units=units, activation=tf.nn.relu)
            # net = tf.layers.dropout(inputs=net, rate=0.5, training=training_phase)
            net = tf.layers.dropout(inputs=net, rate=0.5, training=False)

        # output layer
        q_value = tf.layers.dense(inputs=net, units=len(ActionType))
        prediction = dict()
        # prediction['action'] = tf.multinomial(logits=q_value, num_samples=1)
        prediction['action'] = tf.argmax(input=q_value, axis=1)
        prediction['value'] = q_value
        loss = tf.losses.mean_squared_error(labels=labels['q_value'],
                                            predictions=q_value,
                                            weights=tf.one_hot(indices=labels['action'], depth=len(ActionType), dtype=tf.float32))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return features, labels, prediction, loss, train_op

    @staticmethod
    def making_random_decision():
        return Action(action_type=ActionType(randrange(len(ActionType))))

    @staticmethod
    def normalizing_screen(screen):
        return (screen - 128) / 256

    @staticmethod
    def normalizing_speed(speed):
        return (speed - 125) / 250

    def making_decision(self, screen, speed):
        if random.random() < -0.1:
            print("Choose random action")
            return self.making_random_decision(), np.zeros(shape=[1, len(ActionType)], dtype=np.float32)
        features = {'screen': self.normalizing_screen(screen), 'speed': self.normalizing_speed(speed)}
        prediction = self.continues_evaluation(feature_input=features)
        selected_action = Action(action_type=ActionType(prediction['action']))
        return selected_action, prediction['value']

    def find_state_value(self, screen, speed, batch_size=64):
        state_value = np.zeros([0, len(ActionType)])
        for starting_index in range(0, screen.shape[0], batch_size):
            features = {'screen': self.normalizing_screen(screen[starting_index: min(starting_index + batch_size,
                                                                                     screen.shape[0])]),
                        'speed': self.normalizing_speed(speed[starting_index: min(starting_index + batch_size,
                                                                                  screen.shape[0])])}
            state_value = np.concatenate((state_value, self.continues_evaluation(feature_input=features)['value']), axis=0)
        # return np.amax(state_value, axis=1)
        return state_value

    def training(self, screens, speeds, actions, rewards):
        features = {'screen': self.normalizing_screen(screens), 'speed': self.normalizing_speed(speeds)}
        labels = {'q_value': rewards, 'action': actions}
        return self.train(input_generator=self.input_generator(features, labels, 64))
