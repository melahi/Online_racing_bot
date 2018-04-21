# In the name of God

import mss
import mss.tools
import cv2
import numpy as np
import win32com.client


class ScoreGrabber:
    def __init__(self):
        self.width_image = 20
        self.height_image = 20
        y_position = 152
        x_positions = [875, 897, 919]
        self.digits_position = list()
        for x_position in x_positions:
            self.digits_position.append({'top': y_position,
                                         'left': x_position,
                                         'width': self.width_image,
                                         'height': self.height_image})
        self.screen_shot = mss.mss()
        # Correcting BlueStacks frame
        auto_it = win32com.client.Dispatch("AutoItX3.Control")
        auto_it.WinMove("BlueStacks", "", 0, 0, 950, 700)

    def grab_scores(self):
        grabbed_score = np.ndarray(shape=[len(self.digits_position), self.width, self.height, 3])
        while True:
            for (i, digit_position) in enumerate(self.digits_position):
                grabbed_digit = np.array(self.screen_shot.grab(digit_position))
                grabbed_score[i, :, :, :] = cv2.cvtColor(grabbed_digit, cv2.COLOR_BGRA2RGB)
            yield grabbed_score.copy()

    def get_score_and_display_it(self):
        while True:
            for (i, digit_position) in enumerate(self.digits_position):
                digit_grabbed = np.array(self.screen_shot.grab(digit_position))
                cv2.imshow(str(i), digit_grabbed)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


def main():
    score_grabber = ScoreGrabber()
    score_grabber.get_score_and_display_it()
    # score_grabber.grab_scores()


if __name__ == "__main__":
    main()
