import cv2
import os
import numpy as np

from Action import Action, ActionType
import time


class Experience:
    def __init__(self, screen=None, speed=None, action=None, predicted_reward=None):
        # In Reinforcement Learning terminology, state includes screen and speed
        self.screen = screen
        self.speed = speed
        self.action = action
        self.predicted_rewards = predicted_reward

    def record(self, screen_file_name, speed_file, action_file, predicted_rewards_file):
        cv2.imwrite(screen_file_name, np.reshape(self.screen, newshape=[self.screen.shape[1], self.screen.shape[2]]))
        speed_file.write(str(self.speed[0, 0]) + "\n")
        action_file.write(self.action.to_string() + "\n")
        predicted_rewards_file.write(','.join([str(self.predicted_rewards[0, i])
                                               for i in range(self.predicted_rewards.shape[1])]) + '\n')


class Memory:
    def __init__(self, path="./Memory_collection"):
        self.path = path

    def find_new_slot_for_experiences(self):
        os.makedirs(self.path, exist_ok=True)
        return os.path.join(self.path, str(int(time.time())))

    def record_experiences(self, experiences, length):
        recording_path = self.find_new_slot_for_experiences()
        os.makedirs(recording_path, exist_ok=True)
        speed_file_path = os.path.join(recording_path, "speed_file.txt")
        action_file_path = os.path.join(recording_path, "action_file.txt")
        predicted_reward_file_path = os.path.join(recording_path, "predicted_reward_file.csv")
        speed_file = open(speed_file_path, "w+")
        action_file = open(action_file_path, "w+")
        predicted_reward_file = open(predicted_reward_file_path, "w+")
        for i in range(length):
            screen_file_path = os.path.join(recording_path, "{}.png".format(i))
            experiences[i].record(screen_file_path, speed_file, action_file, predicted_reward_file)

    def find_experiences(self):
        directories = []
        for directory in os.listdir(self.path):
            experiences_path = os.path.join(self.path, directory)
            if not os.path.isdir(experiences_path):
                continue
            directories.append(experiences_path)
        return directories

    @staticmethod
    def remember_experiences(experience_path):
        speed_file = open(os.path.join(experience_path, "speed_file.txt"), mode="r")
        speeds = [float(line.strip()) for line in speed_file]

        # We have a bug that sometime we consider speed as zero when in fact it is 100
        for i in range(1, len(speeds)):
            if speeds[i] < 20:
                speeds[i] = speeds[i - 1]

        action_file = open(os.path.join(experience_path, "action_file.txt"), mode="r")
        actions = [int(line.strip()) for line in action_file]
        screens = [None] * len(speeds)
        for i in range(len(screens)):
            screens[i] = cv2.imread(os.path.join(experience_path, "{}.png".format(i)), flags=cv2.IMREAD_GRAYSCALE)
        return speeds, actions, screens





