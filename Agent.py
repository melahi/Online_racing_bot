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


class Agent:
    def __init__(self):
        self.score_reader = ScoreReader()
        self.screen_grabber = ScreenGrabber()
        self.continue_playing = False
        self.decision_maker = DecisionMaker(screen_width=self.screen_grabber.screen_position['width'],
                                            screen_height=self.screen_grabber.screen_position['height'])
        self.maximum_length_of_experience = 50000
        self.memory = Memory()
        self.look_ahead_step = 25
        self.gamma = 0.98

        self.good_speed = 150
        self.not_bad_speed = 65
        self.bad_speed = 45

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
        if len(self.processed_experience) > 50:
            self.forget_experience(self.processed_experience[9][0])
        
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
                experiences[counter].screen, experiences[counter].speed, self.lowest_reasonable_reward(speed))
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

    def is_successful(self, speeds):
        for speed in speeds:
            if speed < self.good_speed:
                return False
        return True

    def is_failure(self, speeds):
        for i in range(1, len(speeds)):
            if speeds[i - 1] >= self.not_bad_speed and speeds[i] <= self.bad_speed:
                return True
        return False
    
    def load_experience_loss(self):
        self.processed_experience = list()
        with open(self.loss_experience_file, "r") as loss_file:
            for i, line in enumerate(loss_file):
                elements = line.strip().split(sep=",")
                self.processed_experience.append([elements[0], i])
                self.experience_loss.append(float(elements[1]))
    
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
                self.experience_loss.append(1000.0)
        return random.choices(self.processed_experience, weights=self.experience_loss)[0]

    def thinking(self, keep_normal_experience_probability=0.5):
        for _ in range(10):
            experience = self.selecting_an_experience()
            experience_directory = experience[0]
            experience_index = experience[1]
            print("processing {}, with last loss: {}".format(experience_directory,self.experience_loss[experience_index]))
            speeds, actions, screens = self.memory.remember_experiences(experience_path=experience_directory)
            raw_rewards = self.create_rewards(speeds)
            np_screens = np.zeros(shape=[len(raw_rewards), screens[0].shape[0], screens[0].shape[1], 1],
                                  dtype=np.float16)
            np_speeds = np.zeros(shape=[len(raw_rewards), 1], dtype=np.float16)
            np_rewards = np.zeros(shape=[len(raw_rewards), len(ActionType)], dtype=np.float16)
            np_actions = np.zeros(shape=[len(raw_rewards)], dtype=np.int32)
            counter = 0
            trace = plotly.graph_objs.Scatter(y=raw_rewards)
            data = [trace]
            plotly.offline.plot(data, filename="{}-rewards.html".format(experience_directory), auto_open=False)
            for i in range(len(raw_rewards)):
                look_ahead_speeds = speeds[i:i + self.look_ahead_step]
                if (not self.is_successful(look_ahead_speeds) and
                        not self.is_failure(look_ahead_speeds) and
                        random.random() > keep_normal_experience_probability):
                    # This sample is neither a successful sample nor a failure sample, also chance doesn't select it
                    continue

                np_screens[counter, :, :, 0] = screens[i]
                np_speeds[counter, 0] = speeds[i]
                np_actions[counter] = actions[i]
                np_rewards[counter, actions[i]] = raw_rewards[i]
                counter += 1
            print("selects {} samples from {} samples.".format(counter, len(raw_rewards)))
            self.experience_loss[experience_index] = self.decision_maker.training(np_screens[:counter, :, :, :],
                                                                                  np_speeds[:counter, :],
                                                                                  np_actions[:counter],
                                                                                  np_rewards[:counter, :])
            self.save_experience_loss()

    @staticmethod
    def speed_reward(speed):
        if speed <= 50:
            return ((speed - 50) * 5 - 1) * 0.001
        if speed <= 150:
            return ((speed - 100) * 1 + 49) * 0.001
        return ((speed - 150) * 5 + 99) * 0.001

    def create_rewards(self, speeds):
        rewards = []
        for i in range(len(speeds) - self.look_ahead_step - 1):
            # reward = experiences[i].speed
            reward = 0
            gamma_coefficient = 1.0
            for j in range(1, self.look_ahead_step + 1):
                reward += (gamma_coefficient * self.speed_reward(speeds[i + j]))
                gamma_coefficient *= self.gamma
            rewards.append(reward)
        return rewards

    def lowest_reasonable_reward(self, speed):
        lowest_reasonable_speed = speed[0, 0] - 20
        if self.gamma == 1:
            return self.speed_reward(lowest_reasonable_speed) * self.look_ahead_step
        return self.speed_reward(lowest_reasonable_speed) * ((1 - self.gamma ** self.look_ahead_step) /
                                                             (1 - self.gamma))


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
        if need_training:
            agent.thinking()


def simulation():
    agent = Agent()
    agent.simulation()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
    # simulation()
