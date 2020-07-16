from detector_base import Detector
import cv2
from darknet.models import Darknet
from darknet.utils import utils

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

class DarknetDetector(Detector):
    def __init__(self, name, config):
        super(DarknetDetector, self).__init__(name)

        class_path = config.class_path
        yolo_config_path = config.yolo_config_path
        weights_path = config.weights_path
        self.yolo_image_size = config.src_image_size

        classes = utils.load_classes(class_path)

        self.model = Darknet(yolo_config_path, img_size=self.yolo_image_size)
        self.model.load_weights(weights_path)
        self.model.cuda()
        self.model.eval()

        self.conf_thres = config.conf_thres
        self.nms_thres = config.nms_thres

    def get_detection(self, src_image):
        # scale and pad image
        ratio = min(self.yolo_image_size/src_image.size[0], self.yolo_image_size/src_image.size[1])
        imw = round(src_image.size[0] * ratio)
        imh = round(src_image.size[1] * ratio)
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
             transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                            (128,128,128)),
             transforms.ToTensor(),
             ])
        # convert image to Tensor
        image_tensor = img_transforms(src_image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(torch.cuda.FloatTensor))
        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = utils.non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)
        return detections[0]