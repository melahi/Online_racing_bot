import mss
import mss.tools
import cv2
import numpy as np
import win32com.client
import os


class ScreenGrabber:
    def __init__(self):
        primitive_digit_width = 20
        primitive_digit_height = 20
        y_position = 152
        x_positions = [875, 897, 919]
        self.digit_width = 10
        self.digit_height = 10
        self.digits_position = list()
        for x_position in x_positions:
            self.digits_position.append({'top': y_position,
                                         'left': x_position,
                                         'width': primitive_digit_width,
                                         'height': primitive_digit_height})
        self.screen_position = {'top': 210, 'left': 425, 'width': 256, 'height': 128}
        self.screen_shot = mss.mss()
        # Correcting BlueStacks frame
        auto_it = win32com.client.Dispatch("AutoItX3.Control")
        auto_it.WinActivate("BlueStacks", "")
        auto_it.WinMove("BlueStacks", "", 0, 0, 950, 700)

    def grab_scores(self):
        grabbed_digits = [None] * 3
        for (i, digit_position) in enumerate(self.digits_position):
            grabbed_digit = np.array(self.screen_shot.grab(digit_position))
            grabbed_digit = cv2.cvtColor(grabbed_digit, cv2.COLOR_BGRA2GRAY)
            grabbed_digit = cv2.resize(grabbed_digit, (self.digit_width, self.digit_height))
            grabbed_digits[i] = grabbed_digit
        return np.array(grabbed_digits)

    def grab_scores_generator(self):
        while True:
            yield self.grab_scores()

    def get_score_and_display_it(self):
        while True:
            grabbed_digits = self.grab_scores()
            for (i, grabbed_digit) in enumerate(grabbed_digits):
                cv2.imshow(str(i), grabbed_digit)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    @staticmethod
    def reading_images(path):
        features = []
        labels = []
        for class_directory in os.listdir(path):
            for dir_path, _, file_names in os.walk(os.path.join(path, class_directory)):
                for file_name in file_names:
                    features.append(cv2.imread(os.path.join(dir_path, file_name), cv2.IMREAD_GRAYSCALE))
                    labels.append([int(class_directory)])
        return np.asarray(features, dtype=np.float16), np.asarray(labels, dtype=np.int32)

    def grab_screen(self):
        grabbed_screen = np.array(self.screen_shot.grab(self.screen_position))
        grabbed_screen = cv2.cvtColor(grabbed_screen, cv2.COLOR_BGRA2GRAY)
        return np.reshape(grabbed_screen, newshape=[1, grabbed_screen.shape[0], grabbed_screen.shape[1], 1])

    def grab_screen_generator(self):
        while True:
            yield self.grab_screen()


def main():
    score_grabber = ScreenGrabber()
    score_grabber.get_score_and_display_it()


if __name__ == "__main__":
    main()
