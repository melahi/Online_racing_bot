import mss
import mss.tools
import cv2
import numpy as np
import win32com.client
import os


class ScreenGrabber:
    def __init__(self):
        digit_image_width = 20
        digit_image_height = 20
        self.image_width = 10
        self.image_height = 10
        y_position = 152
        x_positions = [875, 897, 919]
        self.digits_position = list()
        for x_position in x_positions:
            self.digits_position.append({'top': y_position,
                                         'left': x_position,
                                         'width': digit_image_width,
                                         'height': digit_image_height})
        self.screen_position = {'top': 50, 'left': 10, 'width': 1072, 'height': 600}
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
            grabbed_digit = cv2.resize(grabbed_digit, (10, 10))
            grabbed_digits[i] = grabbed_digit
        return np.array(grabbed_digits)

    def grab_scores_generator(self):
        while True:
            return self.grab_scores()

    def get_score_and_display_it(self):
        for grabbed_digits in self.grab_scores():
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
                    labels.append(int(class_directory))
        return np.asarray(features, dtype=np.float16), np.asarray(labels, dtype=np.int32)

    def grab_screen(self):
        grabbed_screen = np.array(self.screen_shot.grab(self.screen_position))
        grabbed_screen = cv2.cvtColor(grabbed_screen, cv2.COLOR_BGRA2GRAY)
        return grabbed_screen


def main():
    score_grabber = ScreenGrabber()
    score_grabber.get_score_and_display_it()


if __name__ == "__main__":
    main()
