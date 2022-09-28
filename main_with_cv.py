from imutils.object_detection import non_max_suppression
import cv2 as cv
import numpy as np
import time

from net import setup_mnist_net

FRAME_RATE = 10  # target fps
CAM_ID = 1  # 1 / 0
MNIST_NET = setup_mnist_net("network_weights\\mnist.pth")
PADDING = 20


def timeit(it: callable):
    timer_start = time.time()
    retval = it()
    timer_end = time.time()
    return timer_end - timer_start, retval


def process_image(image):
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, threshed = cv.threshold(grayscale, 100, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
    return threshed


def extract_box_from_image(image, bounding_box):
    (x1, y1), (x2, y2) = bounding_box
    crop = image[y1:y2, x1:x2]
    return crop


def prepare_image_for_mnist_network(image):
    resized = cv.resize(image, (24, 24))
    return cv.copyMakeBorder(resized, 2, 2, 2, 2, cv.BORDER_CONSTANT, value=0)


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
        display_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        processed = process_image(frame)
        contours = cv.findContours(processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)  # bounding box
            # make sure the box is big enough to be significant and that its not the box of the whole image
            if h >= 0.3 * frame_height and h != frame_height:
                p1, p2 = box = (x, y), (x + w, y + h)
                cropped = extract_box_from_image(processed, box)
                prepared = prepare_image_for_mnist_network(cropped)
                prediction = MNIST_NET.run(prepared)
                cv.rectangle(display_frame, p1, p2, color=(0, 255, 0), thickness=2)
                cv.putText(display_frame,
                           text=str(prediction[0][0].item()),
                           org=p1,
                           fontFace=cv.FONT_HERSHEY_PLAIN,
                           fontScale=8,
                           color=(0, 255, 0),
                           thickness=2)
                cv.imshow("preview", cv.resize(prepared, (500, 500), cv.INTER_CUBIC))
                cv.imwrite("pure_cv_preview.png", prepared)

        cv.imshow(self.name, display_frame) # display_frame / processed

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
