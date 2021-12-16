import cv2
import numpy as np

class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img

    # Add your own preprocessing techniques here.

    def grayscale(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def gaussian_threshold(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    def sharpen(self, img):
        kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        return img

    def sharpen2(self, img):
        return cv2.addWeighted(img, 4, cv2.blur(img, (30, 30)), -4, 128)

    def denoise(self, img):
        # Denoising
        return cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
