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
        self.maximum_length_of_experience = 10
        self.memory = Memory()

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
        number_of_samples = 1
        screens = np.zeros(shape=[number_of_samples, 1072, 600, 1], dtype=np.float16)
        speeds = np.zeros(shape=[number_of_samples, 1], dtype=np.float16)
        rewards = np.zeros(shape=[number_of_samples, len(ActionType)], dtype=np.float16)
        actions = np.zeros(shape=[number_of_samples], dtype=np.int32)
        self.decision_maker.training(screens, speeds, actions, rewards)


def main():
    agent = Agent()
    agent.playing(True)
    # agent.thinking()

    reset_action = Action()
    reset_action.apply()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
