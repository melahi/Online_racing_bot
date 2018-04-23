from Action import Action


class DecisionMaker:
    def __init__(self):
        self.max_speed = 90
        pass

    def making_decision(self, screen, speed):
        accelerate = False
        turbo = False
        if speed < self.max_speed and screen[0, 0] > -1:
            accelerate = True

        if speed < 50:
            turbo = True

        return Action(accelerate=accelerate, turbo=turbo, right=True)

