# In the name of God
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework.errors_impl import NotFoundError


class MyEstimator:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.graph = None
        self.feature = None
        self.prediction = None
        self.session = None
        self.is_continues_evaluation_initialized = False
        pass

    def __del__(self):
        self.terminating_continues_evaluation()

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def define_model(self, training_phase):
        feature = None
        label = None
        prediction = None
        loss = None
        train_op = None
        return feature, label, prediction, loss, train_op

    def initialize_model(self, my_session, graph=None):
        try:
            if graph:
                with graph.as_default():
                    saver = tf.train.Saver()
                    saver.restore(my_session, self.model_dir)
            else:
                saver = tf.train.Saver()
                saver.restore(my_session, self.model_dir)
        except NotFoundError:
            if graph:
                with graph.as_default():
                    my_session.run(tf.global_variables_initializer())
            else:
                my_session.run(tf.global_variables_initializer())

    def save_model(self, my_session):
        saver = tf.train.Saver()
        saver.save(my_session, self.model_dir)

    def find_key_and_value(self, dictionary, tensor_key, input_value):
        if type(tensor_key) == dict:
            for key in tensor_key:
                self.find_key_and_value(dictionary, tensor_key[key], input_value[key])
        else:
            dictionary[tensor_key] = input_value

    def train(self, input_generator):
        graph = tf.Graph()
        with graph.as_default():
            feature, label, _, loss, train_op = self.define_model(training_phase=True)
            loss_value = []
            with tf.Session() as my_session:
                self.initialize_model(my_session)
                for counter, (feature_input, label_input) in enumerate(input_generator):
                    model_feed_dict = dict()
                    self.find_key_and_value(model_feed_dict, feature, feature_input)
                    self.find_key_and_value(model_feed_dict, label, label_input)
                    loss_value.append(None)
                    loss_value[-1], _ = my_session.run([loss, train_op], feed_dict=model_feed_dict)
                    if counter % 100 == 0:
                        print("Loss {}: {}".format(len(loss_value), np.mean(loss_value[max(0, counter - 100):])))
                        self.save_model(my_session=my_session)
                print("Final loss {}: {}".format(len(loss_value), np.mean(loss_value)))
                self.save_model(my_session=my_session)
                return np.mean(loss_value)

    def evaluation(self, input_generator):
        graph = tf.Graph()
        with graph.as_default():
            feature, _, prediction, _, _ = self.define_model(training_phase=False)
            with tf.Session() as my_session:
                self.initialize_model(my_session)
                for counter, (feature_input) in enumerate(input_generator):
                    model_feed_dict = dict()
                    self.find_key_and_value(model_feed_dict, feature, feature_input)
                    prediction_value = my_session.run(prediction, feed_dict=model_feed_dict)
                    yield prediction_value

    def initial_continues_evaluation(self):
        if self.is_continues_evaluation_initialized:
            self.terminating_continues_evaluation()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.feature, _, self.prediction, _, _ = self.define_model(training_phase=False)
        self.session = tf.Session(graph=self.graph)
        self.initialize_model(self.session, self.graph)
        self.is_continues_evaluation_initialized = True

    def terminating_continues_evaluation(self):
        if self.session:
            self.session.close()
            del self.session
            self.session = None
        if self.graph:
            del self.graph
            self.graph = None
        self.is_continues_evaluation_initialized = False

    def continues_evaluation(self, feature_input):
        if not self.is_continues_evaluation_initialized:
            print("Initializing continues evaluation")
            self.initial_continues_evaluation()
        model_feed_dict = dict()
        self.find_key_and_value(model_feed_dict, self.feature, feature_input)
        prediction_value = self.session.run(self.prediction, feed_dict=model_feed_dict)
        return prediction_value

    def find_number_of_samples(self, features):
        if type(features) == dict:
            return self.find_number_of_samples(features[next(iter(features))])
        return features.shape[0]

    def assign_next_batch(self, all_data, sampled_index):
        if type(all_data) == dict:
            batched_data = dict()
            for key in all_data:
                batched_data[key] = self.assign_next_batch(all_data[key], sampled_index)
            return batched_data
        return np.asarray([all_data[i] for i in sampled_index])

    def is_nan(self, input_data):
        if type(input_data) == dict:
            for key in input_data:
                if self.is_nan(input_data[key]):
                    return True
            return False
        else:
            return np.isnan(input_data).any()

    def input_generator(self, features, labels, batch_size):
        number_of_samples = self.find_number_of_samples(features)
        assert not self.is_nan(features)
        assert not self.is_nan(labels)

        steps = 1
        for i in range(steps):
            index_order = np.arange(number_of_samples)
            np.random.shuffle(index_order)
            current_index = 0
            while current_index < number_of_samples:
                next_index = min(number_of_samples, current_index + batch_size)
                batch_indices = [index_order[i] for i in range(current_index, next_index)]
                batched_feature = self.assign_next_batch(features, batch_indices)
                batched_label = self.assign_next_batch(labels, batch_indices)
                current_index = next_index
                yield batched_feature, batched_label
            print("Epoch {} is finished".format(i))
