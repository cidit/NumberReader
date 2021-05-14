from imutils.object_detection import non_max_suppression
import cv2 as cv
import numpy as np
import time

from net import setup_mnist_net

FRAME_RATE = 10  # target fps
CAM_ID = 0  # 1
EAST_NET = cv.dnn.readNet("network_weights\\frozen_east_text_detection.pb")
MNIST_NET = setup_mnist_net("network_weights\\mnist.pth")
EAST_OUTPUT_LAYER_NAMES = ["feature_fusion/Conv_7/Sigmoid",  # confidence values
                           "feature_fusion/concat_3", ]  # geometry of the bounding boxes
MINIMUM_BOX_CONFIDENCE = 0.1
PADDING = 20  # px


def timeit(it: callable):
    timer_start = time.time()
    retval = it()
    timer_end = time.time()
    return timer_end - timer_start, retval


def east_forward_pass(cv_image):
    magic_mean_from_tutorial = (123.68, 116.78, 103.94)
    (img_height, img_width, _) = cv_image.shape
    blob = cv.dnn.blobFromImage(cv_image,
                                scalefactor=1.0,
                                size=(img_width, img_height),
                                mean=magic_mean_from_tutorial,
                                swapRB=True,
                                crop=False)
    EAST_NET.setInput(blob)
    (scores, geometries) = EAST_NET.forward(EAST_OUTPUT_LAYER_NAMES)

    return scores, geometries


def mnist_forward_pass(image, box):
    p1, p2 = box
    extracted = extract_box_from_image(image, (p1, p2))
    transformed = prepare_image_for_mnist_network(extracted)
    return MNIST_NET.run(transformed)


def interpret_bounding_boxes(scores, geometry):
    # stole this from a tutorial : https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
    rows, columns = scores.shape[2:4]
    bounding_boxes = []
    confidences = []

    # loop over the number of rows
    for y in range(0, rows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, columns):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < MINIMUM_BOX_CONFIDENCE:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            bounding_boxes.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    filtered_bounding_boxes = non_max_suppression(np.array(bounding_boxes), confidences)
    return filtered_bounding_boxes


def pad_boxes(boxes):
    padded = []
    for (startX, startY, endX, endY) in boxes:
        box = (startX - PADDING, startY - PADDING), (endX + PADDING, endY + PADDING)
        padded.append(box)
    return padded


def extract_box_from_image(image, bounding_box):
    (x1, y1), (x2, y2) = bounding_box
    crop = image[y1:y2, x1:x2]
    return crop


def prepare_image_for_mnist_network(image):
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    resized = cv.resize(grayscale, (28, 28))
    adjusted = adjust_brightness_and_contrast(resized)
    return cv.bitwise_not(adjusted)


def adjust_brightness_and_contrast(image):
    # basically copied from : https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.5  # Simple contrast control
    beta = -10  # Simple brightness control
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            new_image[y, x] = np.clip(alpha * image[y, x] + beta, 0, 255)

    return new_image


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
        delta_time, (confidence_scores, geometry) = timeit(lambda: east_forward_pass(frame))
        raw_boxes = interpret_bounding_boxes(confidence_scores, geometry)
        padded_boxes = pad_boxes(raw_boxes)
        display_frame = frame.copy()

        for box in padded_boxes:
            p1, p2 = box
            cv.rectangle(display_frame, p1, p2, (0, 255, 0), 2)
            dt, prediction = timeit(lambda: mnist_forward_pass(frame, box))
            print("MNIST net forward calculation time:  {:.2f} seconds".format(dt))
            cv.putText(display_frame,
                       text=str(prediction[0][0].item()),
                       org=p1,
                       fontFace=cv.FONT_HERSHEY_PLAIN,
                       fontScale=8,
                       color=(0, 255, 0),
                       thickness=2)

        self.show_preview(frame, padded_boxes)

        print("EAST net forward calculation time:  {:.2f} seconds".format(delta_time))
        cv.imshow(self.name, display_frame)

    def show_preview(self, frame, padded_boxes):
        if len(padded_boxes) > 0:
            extracted = extract_box_from_image(image=frame,
                                               bounding_box=padded_boxes[0])
            transformed = prepare_image_for_mnist_network(extracted)
            cv.imshow("preview", transformed)
            cv.imwrite("east_preview.png", transformed)

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
