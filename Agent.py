# In the name of God
import random
import time
import os

import plotly

from Action import Action, ActionType
from ScoreReader import ScoreReader
from ScreenGrabber import ScreenGrabber
from Memory import Experience, Memory
from DecisionMaker import DecisionMaker
import numpy as np
import tensorflow as tf
import cv2


class Agent:
    def __init__(self):
        self.score_reader = ScoreReader()
        self.screen_grabber = ScreenGrabber()
        self.continue_playing = False
        self.decision_maker = DecisionMaker(screen_width=self.screen_grabber.screen_position['width'],
                                            screen_height=self.screen_grabber.screen_position['height'])
        self.maximum_length_of_experience = 50000
        self.memory = Memory()
        self.gamma = 0.925

        self.processed_experience = list()
        self.experience_loss = list()
        self.loss_experience_file = os.path.join(".\\Memory_collection", "experience_loss.txt")
        self.forgotten_directory = ".\\forgotten_experiences"
        self.load_experience_loss()

    def simulation(self):
        experience_path = "./Simulation_feeding"
        speeds, _, screens = self.memory.remember_experiences(experience_path=experience_path)
        self.memory.path = "./Simulations"
        self.playing(True,
                     np.reshape(speeds, [-1, 1, 1]),
                     np.reshape(screens, [-1, 1, self.decision_maker.screen_height, self.decision_maker.screen_width, 1]),
                     True,
                     False)

    def playing(self, record_experience, score_reader=None, screen_grabber=None, simulation_mode=False):
        print(self.processed_experience)
        if len(self.processed_experience) > 100:
            self.forget_experience(self.processed_experience[0][0])
        
        print("Start new game")
        if score_reader is None:
            score_reader = self.score_reader.read_score()
        if screen_grabber is None:
            screen_grabber = self.screen_grabber.grab_screen_generator()

        if record_experience:
            experiences = [Experience() for _ in range(self.maximum_length_of_experience)]
        else:
            experiences = [Experience()]
        counter = 0
        self.continue_playing = True
        for speed, screen in zip(score_reader, screen_grabber):
            experiences[counter].screen = screen
            experiences[counter].speed = speed
            if speed[0, 0] == 0:
                print("Zero speed is founded")
                speed[0, 0] = 100
            experiences[counter].action, experiences[counter].predicted_rewards = self.decision_maker.making_decision(
                experiences[counter].screen, experiences[counter].speed)
            if not simulation_mode:
                experiences[counter].action.update_current_action_type()
                experiences[counter].action.apply()
            if record_experience:
                counter += 1
            if not simulation_mode and self.is_game_over():
                break
        print("Game is over")
        if record_experience and counter > 50:
            self.memory.record_experiences(experiences, counter)

    def is_game_over(self):
        scores = self.screen_grabber.grab_scores()
        if np.max(scores[2, :, :]) < 50:
            return True
        return False
    
    def wait_to_finish_ads(self):
        for i in range(1):
            self.thinking()

    def continue_to_play(self):
        my_action = Action()
        my_action.press_key("n")
        time.sleep(2)
        my_action.press_key("b")
        time.sleep(2)
        my_action.press_key("c")
        time.sleep(2)
        while self.is_game_over():
            my_action.press_key("s")
        time.sleep(10)

    def load_experience_loss(self):
        self.processed_experience = list()
        with open(self.loss_experience_file, "r") as loss_file:
            for i, line in enumerate(loss_file):
                elements = line.strip().split(sep=",")
                self.processed_experience.append([elements[0], i])
                if elements[1] == "nan":
                    elements[1] = "100000000000.0"
                try:
                    self.experience_loss.append(float(elements[1]))
                except:
                    self.experience_loss.append(100000000000.0)
    
    def save_experience_loss(self):
        with open(self.loss_experience_file, "w") as experience_loss_file:
            for i in range(len(self.processed_experience)):
                experience_loss_file.write("{},{}\n".format(self.processed_experience[i][0], self.experience_loss[i]))
    
    def find_experience_index(self, directory):
        for index, processed_directory in enumerate(self.processed_experience):
            if directory == processed_directory[0]:
                return index 
        return None
    
    def forget_experience(self, experience_directory):
        index = self.find_experience_index(experience_directory)
        if index is None:
            return
        del self.processed_experience[index]
        del self.experience_loss[index]
        os.rename(experience_directory, os.path.join(self.forgotten_directory, experience_directory))
        self.save_experience_loss()
        
    def selecting_an_experience(self):
        directories = self.memory.find_experiences()
        for directory in directories:
            if self.find_experience_index(directory) is None:
                new_index = len(self.processed_experience)
                self.processed_experience.append([directory, new_index])
                self.experience_loss.append(100000000000.0)
        return random.choices(self.processed_experience, weights=self.experience_loss)[0]

    def thinking(self):
        for _ in range(10):
            experience = self.selecting_an_experience()
            experience_directory = experience[0]
            experience_index = experience[1]
            print("processing {}, with last loss: {}".format(experience_directory,
                                                             self.experience_loss[experience_index]))
            speeds, actions, screens = self.memory.remember_experiences(experience_path=experience_directory)
            # raw_rewards = self.create_rewards(speeds)
            samples_count = len(speeds) - 5
            np_screens = np.zeros(shape=[samples_count, screens[0].shape[0], screens[0].shape[1], 1],
                                  dtype=np.float32)
            np_speeds = np.zeros(shape=[samples_count, 1], dtype=np.float32)
            np_rewards = np.zeros(shape=[samples_count, len(ActionType)], dtype=np.float32)
            np_actions = np.zeros(shape=[samples_count], dtype=np.int32)
            for i in range(samples_count):
                np_screens[i, :, :, 0] = screens[i]
                np_speeds[i, 0] = speeds[i]
                np_actions[i] = actions[i]
            raw_rewards = self.create_rewards(np_screens, np_speeds, experience_directory + "_analyzing")
            for i in range(len(raw_rewards)):
                np_rewards[i, actions[i]] = raw_rewards[i]
            print("selects {} samples from {} samples.".format(len(raw_rewards), len(raw_rewards)))
            self.experience_loss[experience_index] =\
                self.decision_maker.training(np_screens[:len(raw_rewards), :, :, :],
                                             np_speeds[:len(raw_rewards), :],
                                             np_actions[:len(raw_rewards)],
                                             np_rewards[:len(raw_rewards), :])
            self.experience_loss[experience_index] = (self.experience_loss[experience_index] / 50) ** 2
            print("Loss score of experience: {}".format(self.experience_loss[experience_index]))
            self.save_experience_loss()

    @staticmethod
    def speed_reward(speed):
        if speed <= 50:
            return ((speed - 50) * 20 - 11) * 0.1
        if speed <= 150:
            return ((speed - 100) * 1 + 39) * 0.1
        return ((speed - 150) * 5 + 89) * 0.1

    def create_rewards(self, screens, speeds, directory):
        os.makedirs(directory, exist_ok=True)
        state_value = self.decision_maker.find_state_value(screens, speeds, 20)
        # dumping state values
        with open(os.path.join(directory, "predicted_state_values.csv"), mode="w") as file:
            for elements in state_value:
                file.write(','.join([str(i) for i in elements]) + '\n')
        # dumping screens
        # counter = 0
        # for screen in screens:
        #     new_shape_screen = np.reshape(screen, newshape=[screen.shape[0], screen.shape[1]])
        #     new_shape_screen = new_shape_screen.astype(np.uint8)
        #     screen_file = os.path.join(directory, "{}.png".format(counter))
        #     cv2.imwrite(screen_file, new_shape_screen)
        #     counter += 1
        # dumping speeds
        with open(os.path.join(directory, "speeds.txt"), mode="w") as file:
            for speed in speeds:
                file.write("{}\n".format(speed))

        max_state_value = np.amax(state_value, axis=1)
        rewards = []
        for i in range(speeds.shape[0] - 1):
            rewards.append(self.speed_reward(speeds[i][0]) + self.gamma * max_state_value[i + 1])

        # ploting rewards
        trace = plotly.graph_objs.Scatter(y=rewards)
        data = [trace]
        reward_file = os.path.join(directory, "rewards.html")
        plotly.offline.plot(data, filename=reward_file, auto_open=False)   
        return rewards


def main():
    agent = Agent()
    need_playing = False
    need_training = True
    # if need_playing:
    #     agent.wait_to_finish_ads()
    while True:
        if need_playing:
            agent.continue_to_play()
            agent.playing(record_experience=True, simulation_mode=False)
            resetting_action = Action()
            resetting_action.apply()
        if need_training:
            agent.thinking()
        break


def simulation():
    agent = Agent()
    agent.simulation()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    # simulation()
