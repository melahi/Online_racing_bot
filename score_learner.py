# In the name of God

import tensorflow as tf
import numpy as np
import os
import cv2
from ScoreGrabber import ScoreGrabber


class ScoreImageReader:
    @staticmethod
    def __reading_images(path):
        features = []
        labels = []
        if path is None:
            print("Error: There is no path for reading images")
        else:
            for filename in os.listdir(path):
                labels.append(filename.split(sep='.png')[0])
                features.append(cv2.imread(os.path.join(path, filename)))

        return np.asarray(features, dtype=np.float16), np.asarray(labels, dtype=np.int32)

    def dataset_from_file(self, path, repeat_count=None):
        features, labels = self.__reading_images(path)
        training_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        return training_dataset.shuffle(30).repeat(repeat_count).batch(100)


class ScoreReader:
    def __init__(self):
        self.number_of_conv_filters = 8
        self.dense_units = 64
        output = "score_reader_{}_{}".format(self.number_of_conv_filters, self.dense_units)
        self.model = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=output)
        self.steps = 100
        self.image_reader = ScoreImageReader()

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

    def training(self, path):
        self.model.train(input_fn=lambda: self.image_reader.dataset_from_file(path), steps=self.steps)
        evaluation = self.model.evaluate(input_fn=lambda: self.image_reader.dataset_from_file(path, 20))
        print(evaluation)

    def predicting(self, path):
        predictions = self.model.predict(input_fn=lambda: self.image_reader.dataset_from_file(path, 1))
        for prediction in predictions:
            print(prediction)

    def predict_and_save(self, generator, saving_path):
        output_directory = ["{}/{}/".format(saving_path, i) for i in range(0, 10)]
        counter = [0] * 10

        # Creating directories for storing images in corresponding class.
        for directory in output_directory:
            if not os.path.exists(directory):
                os.mkdir(directory)

        for images in generator:
            predictions = self.model.predict(input_fn=lambda: tf.data.Dataset.from_tensors(tf.cast(images, tf.float16)))
            for (i, prediction) in enumerate(predictions):
                cv2.imwrite("{}{}.png".format(output_directory[prediction['class']], counter[prediction['class']]),
                            images[i])
                counter[prediction['class']] += 1


def main():
    print("initializing")
    score_reader = ScoreReader()
    # score_reader.training(path="./scores/")
    # score_reader.predicting(path="./scores/")
    score_grabber = ScoreGrabber()
    score_reader.predict_and_save(score_grabber.grab_scores(), "grabbed_score")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
