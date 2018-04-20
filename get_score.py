# In the name of God

from pymouse import PyMouseEvent


class MouseEventHandler(PyMouseEvent):
    def __init__(self):
        super().__init__()

    def click(self, x, y, button, press):
        print("x position: %d, y position: %d, button: %s, press: %s" % (x, y, button, press))



import mss
import mss.tools
import cv2
import time
import numpy as np

class ScoreGrabber:
    def __init__(self):
        self.first_digit_position = {'top': 144, 'left': 806, 'width': 20, 'height': 20}
        self.second_digit_position = {'top': 144, 'left': 826, 'width': 20, 'height': 20}
        self.third_digit_position = {'top': 144, 'left': 846, 'width': 20, 'height': 20}
        self.screen_shot = mss.mss()

    def grab_scores(self):
        counter = 0
        while counter < 500:
            grabbed_screen = self.screen_shot.grab(self.second_digit_position)
            mss.tools.to_png(grabbed_screen.rgb, grabbed_screen.size, output='{}.png'.format(counter))
            time.sleep(0.5)
            counter += 1

    def get_score_and_display_it(self):
        while True:
            first_digit_grabbed = np.array(self.screen_shot.grab(self.first_digit_position))
            second_digit_grabbed = np.array(self.screen_shot.grab(self.second_digit_position))
            third_digit_grabbed = np.array(self.screen_shot.grab(self.third_digit_position))
            cv2.imshow('first', first_digit_grabbed)
            cv2.imshow('second', second_digit_grabbed)
            cv2.imshow('third', third_digit_grabbed)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


def main():
    score_grabber = ScoreGrabber()
    score_grabber.get_score_and_display_it()
    score_grabber.grab_scores()


if __name__ == "__main__":
    main()
