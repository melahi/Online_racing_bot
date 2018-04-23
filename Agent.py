# In the name of God
from Action import Action
from ScoreReader import ScoreReader
from ScreenGrabber import ScreenGrabber
from Memory import Experience, Memory
from DecisionMaker import DecisionMaker


class Agent:
    def __init__(self):
        self.score_reader = ScoreReader()
        self.screen_grabber = ScreenGrabber()
        self.continue_playing = False
        self.decision_maker = DecisionMaker()
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


def main():
    agent = Agent()
    agent.playing(True)
    reset_action = Action()
    reset_action.apply()


if __name__ == "__main__":
    main()
