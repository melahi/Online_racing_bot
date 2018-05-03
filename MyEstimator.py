# In the name of God
import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError


class MyEstimator:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        pass

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def define_model(self, training_phase):
        feature = None
        label = None
        prediction = None
        loss = None
        train_op = None
        return feature, label, prediction, loss, train_op

    def initialize_model(self, my_session):
        try:
            saver = tf.train.Saver()
            saver.restore(my_session, self.model_dir)
        except NotFoundError:
            my_session.run(tf.global_variables_initializer())

    def save_model(self, my_session):
        saver = tf.train.Saver()
        saver.save(my_session, self.model_dir)

    def train(self, input_generator):
        graph = tf.Graph()
        with graph.as_default():
            feature, label, _, loss, train_op = self.define_model(training_phase=True)
            with tf.Session() as my_session:
                self.initialize_model(my_session)
                loss_value = None
                for counter, (feature_input, label_input) in enumerate(input_generator):
                    loss_value, _ = my_session.run([loss, train_op], feed_dict={feature: feature_input,
                                                                                label: label_input})
                    if counter % 100 == 0:
                        print("Loss: {}".format(loss_value))
                print("Final loss: {}".format(loss_value))
                self.save_model(my_session=my_session)

    def evaluation(self, input_generator):
        graph = tf.Graph()
        with graph.as_default():
            feature, _, prediction, _, _ = self.define_model(training_phase=False)
            with tf.Session() as my_session:
                self.initialize_model(my_session)
                for counter, (feature_input) in enumerate(input_generator):
                    prediction_value = my_session.run(prediction, feed_dict={feature: feature_input})
                    yield prediction_value
