from pymouse import PyMouseEvent


class MouseEventHandler(PyMouseEvent):
    def __init__(self):
        super().__init__()

    def click(self, x, y, button, press):
        print("x position: %d, y position: %d, button: %s, press: %s" % (x, y, button, press))


if __name__ == "__main__":
    mouse_handler = MouseEventHandler()
    mouse_handler.run()
