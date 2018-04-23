import win32com.client


class Action:
    def __init__(self, accelerate=False, brake=False, left=False, right=False, turbo=False):
        self.accelerate = accelerate
        self.brake = brake
        self.left = left
        self.right = right
        self.turbo = turbo
        self.auto_it = win32com.client.Dispatch("AutoItX3.Control")
        self.pressing = {True: "down", False: "up"}
        self.key = {"accelerate": "UP", "brake": "DOWN", "left": "LEFT", "right": "RIGHT", "turbo": "LSHIFT"}

    def to_string(self):
        return "{},{},{},{},{}".format(self.accelerate, self.brake, self.left, self.right, self.turbo)

    def from_string(self, string):
        parsed_string = string.split(sep=',')
        self.accelerate = bool(parsed_string[0])
        self.brake = bool(parsed_string[1])
        self.left = bool(parsed_string[2])
        self.right = bool(parsed_string[3])
        self.turbo = bool(parsed_string[4])

    def apply(self):
        print("=======================")
        print("accel: {}".format(self.accelerate))
        print(self.brake)
        print(self.left)
        print(self.right)
        print(self.turbo)
        self.auto_it.Send("{%s %s}" % (self.key["accelerate"], self.pressing[self.accelerate]))
        self.auto_it.Send("{%s %s}" % (self.key["brake"], self.pressing[self.brake]))
        self.auto_it.Send("{%s %s}" % (self.key["left"], self.pressing[self.left]))
        self.auto_it.Send("{%s %s}" % (self.key["right"], self.pressing[self.right]))
        self.auto_it.Send("{%s %s}" % (self.key["turbo"], self.pressing[self.turbo]))
