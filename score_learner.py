# In the name of God

import tensorflow as tf
import numpy as np
import os
import cv2




def reading_images(path):
    features = []
    labels = []
    for filename in os.listdir(path):
        labels.append(filename.split(sep='.png')[0])
        features.append(cv2.imread(os.path.join(path, filename)))
    return np.asarray(features, dtype=np.float16), np.asarray(labels, dtype=np.int32)


def my_dataset(features, labels):
    training_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return training_dataset.shuffle(30).batch(10).repeat()


class ScoreReader:
    def __init__(self):
        self.number_of_conv_filters = 8
        self.dense_units = 50
        self.model = tf.estimator.Estimator(model_fn=self.model_fn, model_dir="./score_reader")
        self.batch_size = 10
        self.steps = 1000
        self.training_dataset = None

    def preparing_training_dataset(self, features, labels):
        self.training_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        self.training_dataset.shuffle(30).batch(self.batch_size)

    def model_fn(self, features, labels, mode):
        net = tf.reshape(features, [-1, 20, 20, 3])
        net = tf.layers.conv2d(inputs=net, filters=self.number_of_conv_filters, kernel_size=[5, 5], padding='same',
                               activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net = tf.reshape(net, [-1, 10 * 10 * self.number_of_conv_filters])
        net = tf.layers.dense(inputs=net, units=self.dense_units, activation=tf.nn.relu)
        net = tf.layers.dropout(inputs=net, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
        logits = tf.layers.dense(inputs=net, units=10)
        predictions = {
            "class": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['class'])
        }
        return tf.estimator.EstimatorSpec(mode=mode, eval_metric_ops=eval_metrics, loss=loss)

    def training(self, features, labels):
        self.training_dataset = self.training_dataset.repeat()
        print(self.training_dataset)
        print(my_dataset(features, labels))
        self.preparing_training_dataset(features, labels)
        self.model.train(input_fn=lambda: self.training_dataset, steps=self.steps)
        evaluation = self.model.evaluate(input_fn=lambda: self.training_dataset.repeat(20))
        print(evaluation)


def main():
    print("initializing")
    features, labels = reading_images("scores")
    score_reader = ScoreReader()
    score_reader.preparing_training_dataset(features, labels)
    score_reader.training(features, labels)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
