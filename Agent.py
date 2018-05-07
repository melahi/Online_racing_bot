# In the name of God
import random
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
        self.maximum_length_of_experience = 1000
        self.memory = Memory()
        self.look_ahead_step = 10
        self.gamma = 0.9

        self.good_speed = 150
        self.not_bad_speed = 65
        self.bad_speed = 45

        self.processed_experience = set()

    def simulation(self):
        all_experiences, _ = self.memory.remember_experiences()
        self.memory.path = "./Simulations"
        for experiences in all_experiences:
            speeds = [np.reshape(np.asarray(experience.speed), newshape=[1, 1]) for experience in experiences]
            screens = [np.reshape(experience.screen,
                                  newshape=[1, experience.screen.shape[0], experience.screen.shape[1], 1])
                       for experience in experiences]
            self.playing(True, speeds, screens, True)

    def playing(self, record_experience, score_reader=None, screen_grabber=None, simulation_mode=False):
        if not score_reader:
            score_reader = self.score_reader.read_score()
        if not screen_grabber:
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
            if speed == 0:
                print("Find zero speed")
            experiences[counter].action, experiences[counter].predicted_rewards = self.decision_maker.making_decision(
                experiences[counter].screen, experiences[counter].speed, self.lowest_reasonable_reward(speed))
            if not simulation_mode:
                experiences[counter].action.apply()
            if record_experience:
                counter += 1
                if counter == self.maximum_length_of_experience:
                    break
        if record_experience:
            self.memory.record_experiences(experiences, counter)

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

    def selecting_an_experience(self):
        directories = self.memory.find_experiences()
        for directory in directories:
            if directory not in self.processed_experience:
                self.processed_experience.add(directory)
                return directory
        return random.choice(directories)

    def thinking(self, keep_normal_experience_probability=0.3):
        while True:
            experience_directory = self.selecting_an_experience()
            print("processing {}".format(experience_directory))
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
            print("selects {} samples from {} samples.".format(len(raw_rewards), counter))
            self.decision_maker.training(np_screens[:counter, :, :, :],
                                         np_speeds[:counter, :],
                                         np_actions[:counter],
                                         np_rewards[:counter, :])

    @staticmethod
    def speed_reward(speed):
        if speed <= 50:
            return (speed - 50) * 5 - 1
        if speed <= 150:
            return (speed - 100) * 1 + 49
        return (speed - 150) * 5 + 99

    def create_rewards(self, speeds):
        rewards = []
        for i in range(len(speeds) - self.look_ahead_step):
            # reward = experiences[i].speed
            reward = 0
            gamma_coefficient = 1.0
            for j in range(1, self.look_ahead_step + 1):
                reward += (gamma_coefficient * self.speed_reward(speeds[i + j]))
                gamma_coefficient *= self.gamma
            rewards.append(reward)
        return rewards

    def lowest_reasonable_reward(self, speed):
        lowest_reasonable_speed = speed[0, 0] - 10
        return self.speed_reward(lowest_reasonable_speed) * ((1 - self.gamma ** self.look_ahead_step) /
                                                             (1 - self.gamma))


def main():
    agent = Agent()
    # agent.playing(record_experience=True, simulation_mode=False)
    agent.thinking()
    # agent.simulation()

    reset_action = Action()
    reset_action.apply()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
