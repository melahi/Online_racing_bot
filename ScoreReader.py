import tensorflow as tf
from hashlib import sha1
import os
import cv2
from ScreenGrabber import ScreenGrabber


class ScoreReader:
    def __init__(self):
        self.dense_units = 200
        output = "score_reader_{}_units".format(self.dense_units)
        self.model = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=output)
        self.steps = 1000
        self.score_grabber = ScreenGrabber()

    def model_fn(self, features, labels, mode):
        net = tf.reshape(features, [-1, 10 * 10 * 1])
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

    @staticmethod
    def create_dataset(features, labels, repeat_count=None):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        return dataset.shuffle(len(features) * 2).repeat(repeat_count).batch(min(10, len(features)))

    def training(self, path):
        features, labels = self.score_grabber.reading_images(path)
        self.model.train(input_fn=lambda: self.create_dataset(features, labels), steps=self.steps)
        evaluation = self.model.evaluate(input_fn=lambda: self.create_dataset(features, labels, 20))
        print(evaluation)

    def predict_and_save(self, generator, saving_path):
        output_directory = ["{}/{}/".format(saving_path, i) for i in range(0, 10)]

        # Creating directories for storing images in corresponding class.
        for directory in output_directory:
            os.makedirs(directory, exist_ok=True)

        for images in generator:
            predictions = self.model.predict(input_fn=lambda: tf.data.Dataset.from_tensors(tf.cast(images, tf.float16)))
            for (i, prediction) in enumerate(predictions):
                image_name = sha1(images[i].tostring()).hexdigest()
                cv2.imwrite("{}{}.png".format(output_directory[prediction['class']], image_name), images[i])

    def read_score(self):
        images = self.score_grabber.grab_scores()
        predictions = self.model.predict(input_fn=lambda: tf.data.Dataset.from_tensors(tf.cast(images, tf.float16)))
        score = 0
        for (i, prediction) in enumerate(predictions):
            score += (10 ** (2 - i)) * prediction['class']
        return score


def main():
    print("initializing")
    score_reader = ScoreReader()
    score_reader.training(path="./grabbed_score/")
    # score_grabber = ScreenGrabber()
    # score_reader.predict_and_save(score_grabber.grab_scores_generator(), "grabbed_score_new/")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
