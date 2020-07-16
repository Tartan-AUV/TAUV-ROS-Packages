from detector_base import Detector
import cv2 as cv
import numpy as np

class TemplateMatchingDetector(Detector):
    def __init__(self, name, template):
        super(TemplateMatchingDetector, self).__init__(name)
        self.template = template

    def get_detection(self, src_image):
        img = src_image
        img2 = img.copy()
        template = self.template
        w, h = template.shape[::-1]
        methods = ['cv.TM_CCOEFF_NORMED']
        for meth in methods:
            img = img2.copy()
            resize_i = img2.copy()
            method = eval(meth)
            orig_res = None
            for i in range(2):
                resize_i = cv.resize(img, None,fx=1/2**(0.5*i), fy=1/2**(0.5*i), interpolation = cv.INTER_AREA)

                # Apply template Matching
                res = cv.matchTemplate(resize_i, template, method)
                if i == 0:
                    orig_res = res

                    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                    threshold = 0.68
                    loc = np.where( res >= threshold)

                    for pt in zip(*loc[::-1]):
                        cv.rectangle(src_image, (pt[0]*int(2**(0.5*i)),pt[1]*int(2**(0.5*i))), ((pt[0] + w), (pt[1] + h)), (0,0,255), 1)

                        cv.imshow('Matching Result', orig_res)
                        cv.imshow('Detected Point', src_image)
