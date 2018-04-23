import cv2
import os

from Action import Action


class Experience:
    def __init__(self, screen=None, speed=None, action=None):
        # In Reinforcement Learning terminology, state includes screen and speed
        self.screen = screen
        self.speed = speed
        self.action = action

    def record(self, screen_file_name, speed_file, action_file):
        cv2.imwrite(screen_file_name, self.screen)
        speed_file.write(str(self.speed) + "\n")
        action_file.write(self.action.to_string() + "\n")


class Memory:
    def __init__(self, path="./Memory_collection"):
        self.path = path

    def find_new_slot_for_experiences(self):
        os.makedirs(self.path, exist_ok=True)
        counter = 0
        new_slot = self.path
        while os.path.exists(new_slot):
            new_slot = os.path.join(self.path, str(counter))
            counter += 1
        return new_slot

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
        for directory in os.listdir(self.path):
            experiences_path = os.path.join(self.path, directory)
            speed_file = open(os.path.join(experiences_path, "speed_file.txt"), mode="r")
            speeds = [int(line.strip()) for line in speed_file]
            action_file = open(os.path.join(experiences_path, "action_file.txt"), mode="r")
            actions = [Action() for _ in range(len(speeds))]
            for i, line in enumerate(action_file):
                actions[i].from_string(line)
            images = []
            for i in range(len(speeds)):
                images.append(cv2.imread(os.path.join(experiences_path, "{}.png".format(i))))

            for i in range(len(speeds)):
                experiences.append(Experience(screen=images[i], speed=speeds[i], action=actions[i]))

        return experiences





