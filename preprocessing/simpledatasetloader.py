import numpy as np
import cv2
import os

class SimpleDatasetLoader(object):
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        for (i, imgPath) in enumerate(imagePaths):
            """
            load image and extract class label
            assumes image has following format:
            /path/to/dataset/{class}/{image}.jpg
            """

            image = cv2.imread(imgPath)
            label = imgPath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] Processed {}/{}".format(i+1, len(imagePaths)))

        return np.array(data), np.array(labels)