from random import randrange

from Action import Action, ActionType
from MyEstimator import MyEstimator
import tensorflow as tf
import os


class DecisionMaker(MyEstimator):
    def __init__(self, screen_width, screen_height):
        self.conv_layers_kernel_size = [3, 3, 3]
        self.conv_layers_filters = [32, 32, 64]
        self.dense_units = [1000, 500, 200]
        output_dir = "decision_maker"
        for i in range(len(self.conv_layers_filters)):
            output_dir += "_conv_{}_{}".format(self.conv_layers_kernel_size[i], self.conv_layers_filters[i])
        for units in self.dense_units:
            output_dir += "_dense{}".format(units)
        os.makedirs(output_dir, exist_ok=True)
        model_dir = os.path.join(output_dir, "model.ckpt")
        super().__init__(model_dir=model_dir)
        self.down_sample_factor = 2 ** len(self.conv_layers_filters)
        assert (screen_width % self.down_sample_factor == 0), "screen_width should be divisible by {}.".\
            format(self.down_sample_factor)
        assert (screen_height % self.down_sample_factor == 0), "screen_height should be divisible by {}.".\
            format(self.down_sample_factor)
        self.screen_width = screen_width
        self.screen_height = screen_height

    def define_model(self, training_phase):
        features = dict()
        labels = dict()
        features['screen'] = tf.placeholder(dtype=tf.float16, shape=[None, self.screen_height, self.screen_width, 1])
        features['speed'] = tf.placeholder(dtype=tf.float16, shape=[None, 1])
        labels['q_value'] = tf.placeholder(dtype=tf.float16, shape=[None, len(ActionType)])
        labels['action'] = tf.placeholder(dtype=tf.int32, shape=[None])
        # net = tf.cast(features['screen'], dtype=tf.float16)
        net = features['screen']
        for i in range(len(self.conv_layers_kernel_size)):
            net = tf.layers.conv2d(inputs=net,
                                   kernel_size=self.conv_layers_kernel_size[i],
                                   filters=self.conv_layers_filters[i],
                                   padding='same',
                                   activation=tf.nn.relu)
            net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.reshape(net, shape=[-1, int(self.screen_width / self.down_sample_factor *
                                             self.screen_height / self.down_sample_factor *
                                             self.conv_layers_filters[-1])])
        net = tf.concat([net, features['speed']], axis=1)
        for units in self.dense_units:
            net = tf.layers.dense(inputs=net, units=units, activation=tf.nn.relu)

        # output layer
        q_value = tf.layers.dense(inputs=net, units=len(ActionType))

        prediction = tf.argmax(input=q_value, axis=1)
        loss = tf.losses.mean_squared_error(labels=labels['q_value'],
                                            predictions=q_value,
                                            weights=tf.one_hot(indices=labels['action'], depth=len(ActionType)))

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return features, labels, prediction, loss, train_op

    @staticmethod
    def making_random_decision():
        return Action(action_type=ActionType(randrange(len(ActionType))))

    def making_decision(self, screen, speed):
        # return self.making_random_decision()
        features = {'screen': screen, 'speed': speed}
        prediction = self.continues_evaluation(feature_input=features)
        return Action(action_type=ActionType(prediction))

    def training(self, screens, speeds, actions, rewards):
        features = {'screen': screens, 'speed': speeds}
        labels = {'q_value': rewards, 'action': actions}
        self.train(input_generator=self.input_generator(features, labels, 10))
