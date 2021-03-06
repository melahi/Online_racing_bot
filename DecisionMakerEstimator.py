from random import randrange

from Action import Action, ActionType
import tensorflow as tf


class DecisionMaker:
    def __init__(self):
        self.steps = 1000
        self.conv_layers_kernel_size = [8, 4]
        self.conv_layers_stride_size = [4, 2]
        self.conv_layers_filters = [16, 32]
        self.dense_units = [256]
        output = "decision_maker"
        for i in range(len(self.conv_layers_filters)):
            output += "_conv_{}_{}".format(self.conv_layers_kernel_size[i], self.conv_layers_filters[i])
        for units in self.dense_units:
            output += "_dense{}".format(units)
        self.model = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=output)

    def model_fn(self, features, labels, mode):
        net = tf.cast(features['screen'], dtype=tf.float32)
        for i in range(len(self.conv_layers_kernel_size)):
            net = tf.layers.conv2d(inputs=net,
                                   kernel_size=self.conv_layers_kernel_size[i],
                                   filters=self.conv_layers_filters[i],
                                   strides=self.conv_layers_stride_size[i],
                                   padding='same',
                                   activation=tf.nn.relu)
        net = tf.layers.flatten(inputs=net)
        net = tf.concat([net, features['speed']], axis=1)
        for units in self.dense_units:
            net = tf.layers.dense(inputs=net, units=units, activation=tf.nn.relu)

        # output layer
        q_value = tf.layers.dense(inputs=net, units=len(ActionType))

        predictions = {
            "action_type": tf.argmax(input=q_value, axis=1),
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.mean_squared_error(labels=labels['q_value'],
                                            predictions=q_value,
                                            weights=tf.one_hot(indices=labels['action'], depth=len(ActionType)))

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        print("mode == tf.estimator.ModeKeys.EVAL was not supporting for decision making")
        exit(9)

    @staticmethod
    def making_random_decision():
        return Action(action_type=ActionType(randrange(len(ActionType))))

    def making_decision(self, screen, speed):
        features = {'screen': screen, 'speed': speed}
        predictions = self.model.predict(input_fn=lambda: tf.data.Dataset.from_tensors(features))
        return Action(action_type=ActionType(next(predictions)['action_type']))

    def training(self, screens, speeds, actions, rewards):
        features = {'screen': screens, 'speed': speeds}
        labels = {'q_value': rewards, 'action': actions}
        self.model.train(input_fn=lambda: self.create_dataset(features, labels), steps=self.steps)

    @staticmethod
    def create_dataset(features, labels, repeat_count=None):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        return dataset.shuffle(len(features) * 2).repeat(repeat_count).batch(min(40, len(features)))

