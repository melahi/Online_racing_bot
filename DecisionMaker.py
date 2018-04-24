from Action import Action
import tensorflow as tf


class DecisionMaker:
    def __init__(self, screen_width, screen_height):
        self.conv_layers_kernel_size = [3, 3, 3]
        self.conv_layers_filters = [16, 32, 64]
        self.dense_units = [1024, 512]
        output = "decision_maker"
        for i in range(len(self.conv_layers_filters)):
            output += "_conv_{}*{}".format(self.conv_layers_kernel_size[i], self.conv_layers_filters[i])
        for units in self.dense_units:
            output += "_dense{}".format(units)
        self.model = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=output)
        self.down_sample_factor = 2 ** len(self.conv_layers_filters)
        assert (screen_width % self.down_sample_factor == 0), "screen_width should be divisible by {}.".\
            format(self.down_sample_factor)
        assert (screen_height % self.down_sample_factor == 0), "screen_height should be divisible by {}.".\
            format(self.down_sample_factor)
        self.screen_width = screen_width
        self.screen_height = screen_height


    def model_fn(self, features, labels, mode):
        net = features
        for i in range(len(self.conv_layers_kernel_size)):
            net = tf.layers.conv2d(inputs=net,
                                   kernel_size=self.conv_layers_kernel_size[i],
                                   filters=self.conv_layers_filters[i],
                                   padding='same',
                                   activation=tf.nn.relu)
            net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.reshape(net, shape=[-1, (self.screen_width / self.down_sample_factor *
                                          self.screen_height / self.down_sample_factor *
                                          self.conv_layers_filters[-1])])
        for units in self.dense_units:
            net = tf.layers.dense(inputs=net, units=units, activation=tf.nn.relu)

        # output layer
        net = tf.layers.dense(input=net, units = 5, activation=tf.nn.sigmoid)


    def making_decision(self, screen, speed):
        accelerate = False
        turbo = False
        if speed < 90 and screen[0, 0] > -1:
            accelerate = True

        if speed < 50:
            turbo = True

        return Action(accelerate=accelerate, turbo=turbo, right=True)

