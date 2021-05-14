import cv2 as cv
import pytesseract as tesseract

FRAME_RATE = 5  # fps
CAM_ID = 1
TESSERACT_PATH = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

tesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


class App:
    name: str
    shortcuts: dict[str, callable]

    def __init__(self, name: str):
        """
        Constructor
        """
        self.name = name
        self.shortcuts = {
            'h': self.help,
        }

        cv.namedWindow(name)

    def help(self):
        print(f"help text! for {self.name}")

    def run(self):
        """
        starts event loop
        """
        print(f"Running App: {self.name}")

        delay = int(1000 / FRAME_RATE)
        capture = cv.VideoCapture(CAM_ID)

        while True:
            capture_success, frame = capture.read()
            if capture_success:
                cv.imshow(self.name, frame)
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                data = tesseract.image_to_string(rgb_frame)
                print(data)
                if (k := cv.waitKey(delay)) != -1:
                    key = chr(k).lower()
                    if key == 'q':
                        break
                    elif key != 'q':
                        if key in self.shortcuts.keys():
                            self.shortcuts[key]()
                        else:
                            print(f"key entered: {key}")

        capture.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    app = App("Number Reader")
    app.run()
