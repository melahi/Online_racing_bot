import win32com.client
from enum import Enum


class ActionType(Enum):
    NOTHING = 0
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    UP_RIGHT = 5
    UP_LEFT = 6
    UP_TURBO = 7
    DOWN_RIGHT = 8
    DOWN_LEFT = 9
    UP_RIGHT_TURBO = 10
    UP_LEFT_TURBO = 11


class Action:
    def __init__(self, action_type=ActionType.NOTHING):
        self.action_to_key = [None] * len(ActionType)
        self.auto_it = win32com.client.Dispatch("AutoItX3.Control")
        self.pressing = {True: "down", False: "up"}
        self.action_type = action_type
        self.prepare_action_type_to_key_status_dict()

    def prepare_action_type_to_key_status_dict(self):
        for action_type in ActionType:
            key_status = list()

            # UP
            if action_type in [ActionType.UP, ActionType.UP_RIGHT, ActionType.UP_LEFT, ActionType.UP_TURBO,
                               ActionType.UP_RIGHT_TURBO, ActionType.UP_LEFT_TURBO]:
                key_status.append("down")
            else:
                key_status.append("up")

            # Down
            if action_type in [ActionType.DOWN, ActionType.DOWN_RIGHT, ActionType.DOWN_LEFT]:
                key_status.append("down")
            else:
                key_status.append("up")

            # RIGHT
            if action_type in [ActionType.RIGHT, ActionType.UP_RIGHT, ActionType.UP_RIGHT_TURBO, ActionType.DOWN_RIGHT]:
                key_status.append("down")
            else:
                key_status.append("up")

            # LEFT
            if action_type in [ActionType.LEFT, ActionType.UP_LEFT, ActionType.UP_LEFT_TURBO, ActionType.DOWN_LEFT]:
                key_status.append("down")
            else:
                key_status.append("up")

            # TURBO
            if action_type in [ActionType.UP_TURBO, ActionType.UP_RIGHT_TURBO, ActionType.UP_LEFT_TURBO]:
                key_status.append("down")
            else:
                key_status.append("up")

            self.action_to_key[action_type.value] = key_status

    def to_string(self):
        return str(self.action_type.value)

    def from_string(self, serialized_action_type):
        self.action_type = ActionType(int(serialized_action_type))

    def apply(self):
        print("=======================")
        print("{UP %s}" % self.action_to_key[self.action_type.value][0])
        print("{DOWN %s}" % self.action_to_key[self.action_type.value][1])
        print("{RIGHT %s}" % self.action_to_key[self.action_type.value][2])
        print("{LEFT %s}" % self.action_to_key[self.action_type.value][3])
        print("{TURBO %s}" % self.action_to_key[self.action_type.value][4])

        self.auto_it.Send("{UP %s}" % self.action_to_key[self.action_type.value][0])
        self.auto_it.Send("{DOWN %s}" % self.action_to_key[self.action_type.value][1])
        self.auto_it.Send("{RIGHT %s}" % self.action_to_key[self.action_type.value][2])
        self.auto_it.Send("{LEFT %s}" % self.action_to_key[self.action_type.value][3])
        self.auto_it.Send("{TURBO %s}" % self.action_to_key[self.action_type.value][4])
