import cv2
import os
import numpy as np

from Action import Action, ActionType
import time


class Experience:
    def __init__(self, screen=None, speed=None, action=None):
        # In Reinforcement Learning terminology, state includes screen and speed
        self.screen = screen
        self.speed = speed
        self.action = action

    def record(self, screen_file_name, speed_file, action_file):
        cv2.imwrite(screen_file_name, np.reshape(self.screen, newshape=[self.screen.shape[1], self.screen.shape[2]]))
        speed_file.write(str(self.speed[0, 0]) + "\n")
        action_file.write(self.action.to_string() + "\n")


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
        speed_file = open(speed_file_path, "w+")
        action_file = open(action_file_path, "w+")
        for i in range(length):
            screen_file_path = os.path.join(recording_path, "{}.png".format(i))
            experiences[i].record(screen_file_path, speed_file, action_file)

    def remember_experiences(self):
        experiences = []
        directories = []
        for directory in os.listdir(self.path):
            experiences_path = os.path.join(self.path, directory)
            if not os.path.isdir(experiences_path):
                continue
            experiences.append([])
            directories.append(experiences_path)
            speed_file = open(os.path.join(experiences_path, "speed_file.txt"), mode="r")
            speeds = [float(line.strip()) for line in speed_file]
            action_file = open(os.path.join(experiences_path, "action_file.txt"), mode="r")
            actions = [Action(action_type=ActionType(int(line.strip()))) for line in action_file]
            images = []
            for i in range(len(speeds)):
                images.append(cv2.imread(os.path.join(experiences_path, "{}.png".format(i)),
                                         flags=cv2.IMREAD_GRAYSCALE))

            for i in range(len(speeds)):
                experiences[-1].append(Experience(screen=images[i], speed=speeds[i], action=actions[i]))
        return experiences, directories





