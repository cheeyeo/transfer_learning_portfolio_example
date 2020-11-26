import imutils
import cv2

class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        h, w = image.shape[:2]
        dW = 0
        dH = 0

        # if w < h, resize along the width; update deltas to crop along the height
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            # h < w so resize along height; update deltas to crop along the width
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        # regrab width and height and do crop
        (h, w) = image.shape[:2]
        image = image[dH:h-dH, dW:w-dW]

        # resize image to provided spatial dimensions so output image is always a fixed size
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)