import tensorflow as tf
from MyEstimator import MyEstimator
from hashlib import sha1
import os
import cv2
from ScreenGrabber import ScreenGrabber
import numpy as np


class ScoreReader(MyEstimator):
    def __init__(self):
        self.dense_units = 200
        model_dir = os.path.join("score_reader_{}_units".format(self.dense_units), "model.ckpt")
        super().__init__(model_dir=model_dir)
        self.steps = 1000
        self.screen_grabber = ScreenGrabber()

    def define_model(self, training_phase):
        features = tf.placeholder(dtype=tf.float16, shape=[None,
                                                           self.screen_grabber.digit_width,
                                                           self.screen_grabber.digit_height])
        labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        net = tf.reshape(features, shape=[-1, self.screen_grabber.digit_width * self.screen_grabber.digit_height])
        net = tf.layers.dense(inputs=net, units=self.dense_units, activation=tf.nn.relu)
        net = tf.layers.dropout(inputs=net, rate=0.4, training=training_phase)
        logits = tf.layers.dense(inputs=net, units=10)
        prediction = tf.argmax(input=logits, axis=1)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return features, labels, prediction, loss, train_op

    def training_phase(self, path):
        def input_generator(input_path, batch_size):
            features, labels = ScreenGrabber.reading_images(path=input_path)
            number_of_samples = features.shape[0]
            for _ in range(self.steps):
                index_order = np.arange(number_of_samples)
                np.random.shuffle(index_order)
                current_index = 0
                while current_index < number_of_samples:
                    next_index = min(number_of_samples, current_index + batch_size)
                    batched_feature = np.asarray([features[index_order[i]] for i in range(current_index,
                                                                                          next_index)])
                    batched_label = np.asarray([[labels[index_order[i]]] for i in range(current_index,
                                                                                        next_index)])
                    current_index = next_index
                    yield batched_feature, batched_label
        self.train(input_generator=input_generator(input_path=path, batch_size=50))

    def read_score(self):
        predictions_gen = self.evaluation(input_generator=self.screen_grabber.grab_scores_generator())
        for predictions in predictions_gen:
            score = np.zeros(shape=[1, 1], dtype=np.float16)
            for (i, prediction) in enumerate(predictions):
                score[0, 0] += (10 ** (2 - i)) * prediction
            yield score

    def predict_and_save(self, saving_path):
        output_directory = ["{}/{}/".format(saving_path, i) for i in range(0, 10)]

        # Creating directories for storing images in corresponding class.
        for directory in output_directory:
            os.makedirs(directory, exist_ok=True)

        for images in self.screen_grabber.grab_scores_generator():
            images = [images]
            predictions = self.evaluation(input_generator=images)
            for (i, prediction) in enumerate(next(predictions)):
                print(prediction)
                image_name = sha1(images[0][i].tostring()).hexdigest()
                print("{}{}.png".format(output_directory[prediction], image_name))
                cv2.imwrite("{}{}.png".format(output_directory[prediction], image_name), images[0][i])


def main():
    print("initializing")
    score_reader = ScoreReader()
    # score_reader.predict_and_save(saving_path="my_test")
    # score_reader.training_phase(path="./grabbed_score/")
    for i in score_reader.read_score():
        print(i)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
