from imutils.object_detection import non_max_suppression
import cv2 as cv
import numpy as np
import time

from net import setup_mnist_net

FRAME_RATE = 10  # target fps
CAM_ID = 0  # 1
MNIST_NET = setup_mnist_net("network_weights\\mnist.pth")


def timeit(it: callable):
    timer_start = time.time()
    retval = it()
    timer_end = time.time()
    return timer_end - timer_start, retval


class App:
    name: str
    shortcuts: dict[str, callable]
    capture: cv.VideoCapture
    running: bool = True

    def __init__(self, name: str):
        """
        Constructor
        """
        self.name = name
        self.shortcuts = {
            'h': self.help,
            'q': self.quit,
        }

        self.capture = cv.VideoCapture(CAM_ID)
        cv.namedWindow(name)

    def help(self):
        print(f"help text! for {self.name}")

    def quit(self):
        self.running = False

    def run(self):
        """
        event loop
        """
        print(f"Running App: {self.name}")
        delay = int(1000 / FRAME_RATE)

        while self.running:
            process_time, _ = timeit(lambda: self.process_next_frame())
            if (key := cv.waitKey(int(delay - process_time))) != -1:
                self.handle_shortcut_event(chr(key).lower())

        self.cleanup()

    def process_next_frame(self):
        capture_success, frame = self.capture.read()
        if capture_success:
            self.handle_capture(frame)
        else:
            print("failed to capture")

    def handle_capture(self, frame):
        cv.imshow(self.name, frame)

    def handle_shortcut_event(self, shortcut):
        if shortcut in self.shortcuts.keys():
            self.shortcuts[shortcut]()
        else:
            print(f"invalid shortcut entered: {shortcut}")

    def cleanup(self):
        self.capture.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    app = App("Number Reader")
    app.run()
