# In the name of God
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
        self.maximum_length_of_experience = 500
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
        while self.continue_playing:
            experiences[counter].screen = self.screen_grabber.grab_screen()
            experiences[counter].speed = self.score_reader.read_score()
            experiences[counter].action = self.decision_maker.making_decision(experiences[counter].screen,
                                                                              experiences[counter].speed)
            experiences[counter].action.apply()
            if record_experience:
                counter += 1
                if counter == self.maximum_length_of_experience:
                    self.memory.record_experiences(experiences, counter)
                    self.continue_playing = False

    def thinking(self):
        experiences = self.memory.remember_experiences()
        raw_rewards = self.create_rewards(experiences=experiences)

        # Removing experience which their corresponding rewards is not defined
        samples_count = len(raw_rewards)
        experiences = experiences[:samples_count]

        screens = np.zeros(shape=[samples_count, experiences[0].screen.shape[0], experiences[1].screen.shape[1], 1],
                             dtype=np.float16)
        speeds = np.zeros(shape=[samples_count, 1], dtype=np.float16)
        rewards = np.zeros(shape=[samples_count, len(ActionType)], dtype=np.float16)
        actions = np.zeros(shape=[samples_count], dtype=np.int32)
        for i in range(samples_count):
            screens[i, :, :, 0] = experiences[i].screen
            speeds[i, 0] = experiences[i].speed
            actions[i] = experiences[i].action.action_type.value
            rewards[i, actions[i]] = raw_rewards[i]
        self.decision_maker.training(screens, speeds, actions, rewards)

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
    agent.playing(True)
    # agent.thinking()

    reset_action = Action()
    reset_action.apply()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
