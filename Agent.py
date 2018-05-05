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

    def playing(self, record_experience):
        if record_experience:
            experiences = [Experience() for _ in range(self.maximum_length_of_experience)]
        else:
            experiences = [Experience()]
        counter = 0
        self.continue_playing = True
        for speed in self.score_reader.read_score():
            experiences[counter].screen = self.screen_grabber.grab_screen()
            experiences[counter].speed = speed
            experiences[counter].action = self.decision_maker.making_decision(experiences[counter].screen,
                                                                              experiences[counter].speed)
            experiences[counter].action.apply()
            if record_experience:
                counter += 1
                if counter == self.maximum_length_of_experience:
                    self.memory.record_experiences(experiences, counter)
                    break

    def thinking(self, threshold=None, sample_ignoring_probability=1.0):
        all_experiences, directories = self.memory.remember_experiences()
        all_raw_rewards = []
        samples_count = 0
        for i in range(len(all_experiences)):
            all_raw_rewards.append(self.create_rewards(experiences=all_experiences[i]))
            samples_count += len(all_raw_rewards[-1])
            # Removing experience which their corresponding rewards is not defined
            all_experiences[i] = all_experiences[i][:samples_count]

        screens = np.zeros(shape=[samples_count,
                                  all_experiences[0][0].screen.shape[0],
                                  all_experiences[0][1].screen.shape[1],
                                  1],
                           dtype=np.float16)
        speeds = np.zeros(shape=[samples_count, 1], dtype=np.float16)
        rewards = np.zeros(shape=[samples_count, len(ActionType)], dtype=np.float16)
        actions = np.zeros(shape=[samples_count], dtype=np.int32)
        counter = 0
        for raw_rewards, experiences, directory in zip(all_raw_rewards, all_experiences, directories):
            trace = plotly.graph_objs.Scatter(y=raw_rewards)
            data = [trace]
            plotly.offline.plot(data, filename="{}-rewards.html".format(directory))
            for raw_reward, experience in zip(raw_rewards, experiences):
                if threshold and raw_reward < threshold:
                    if random.random() < sample_ignoring_probability:
                        continue
                screens[counter, :, :, 0] = experience.screen
                speeds[counter, 0] = experience.speed
                actions[counter] = experience.action.action_type.value
                rewards[counter, actions[counter]] = raw_reward
                counter += 1
        print("{} and {}".format(samples_count, counter))
        self.decision_maker.training(screens[:counter, :, :, :],
                                     speeds[:counter, :],
                                     actions[:counter],
                                     rewards[:counter, :])

    def create_rewards(self, experiences):
        rewards = []
        for i in range(len(experiences) - self.look_ahead_step):
            reward = 0
            gamma_coefficient = 1
            for j in range(1, self.look_ahead_step + 1):
                reward += (gamma_coefficient * (experiences[i + j].speed - experiences[i + j - 1].speed))
                gamma_coefficient *= self.gamma
            rewards.append(reward)
        return rewards


def main():
    agent = Agent()
    # agent.playing(True)
    agent.thinking(threshold=5, sample_ignoring_probability=1.0)

    reset_action = Action()
    reset_action.apply()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
