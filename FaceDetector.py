from __future__ import print_function
import numpy as np
import torch

from face_detection.PriorBox import PriorBox
from face_detection.functions import load_model
from face_detection.faceboxes import FaceBoxes


class FaceDetector:
    def __init__(self):

        self.min_sizes = [[32, 64, 128], [256], [512]]
        self.steps = [32, 64, 128]
        self.variance = [0.1, 0.2]
        self.clip = False
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.3
        self.keep_top_k_pred = 100

        model_path = r"model/faceboxes-pytorch.pth"
        network = FaceBoxes(phase='test', size=None, num_classes=2)
        self.inferrer = load_model(network, model_path)
        self.inferrer.eval()
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            # cudnn.benchmark = True
            self.inferrer = self.inferrer.to(torch.device("cuda"))

    def detect_face(self, frame: np.ndarray) -> list:
        scale = torch.FloatTensor([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
        #
        # if self.use_cpu:
        #     # image preprocessing
        #     frame_to_infer = cv2.resize(frame, (self.input_width, self.input_height))
        #     frame_to_infer = np.expand_dims(frame_to_infer, axis=0)
        #     frame_to_infer = np.transpose(frame_to_infer, axes=(0, 3, 1, 2))
        #     # infer image
        #     result = self.inferrer.infer({self.input_blob: frame_to_infer})
        #     boxes = result['boxes'][0]
        #     scores = result['scores'][0][:, 1]
        #     # post-processing
        #     priorbox = PriorBox(self.min_sizes, self.steps, self.clip, image_size=(self.input_height, self.input_width))
        #     priors = priorbox.forward()
        #     priors = priors.to('cpu')
        #     prior_data = priors.data
        #     boxes = self.decode(torch.from_numpy(np.asarray(boxes.data)), prior_data, priors)
        #     boxes = boxes * scale
        #     boxes = boxes.cpu().numpy()
        # else:
        # image preprocessing
        frame_to_infer = np.float32(frame)
        im_height, im_width, _ = frame_to_infer.shape
        frame_to_infer -= (104, 117, 123)
        frame_to_infer = frame_to_infer.transpose(2, 0, 1)
        frame_to_infer = torch.from_numpy(frame_to_infer).unsqueeze(0)
        if self.use_gpu:
            frame_to_infer = frame_to_infer.to(torch.device("cuda"))
            scale = scale.to(torch.device("cuda"))
        # infer image
        loc, conf = self.inferrer(frame_to_infer)
        # post-processing
        priorbox = PriorBox(self.min_sizes, self.steps, self.clip, image_size=(im_height, im_width))
        priors = priorbox.forward()
        if self.use_gpu:
            priors = priors.to('cuda:0')
        prior_data = priors.data
        boxes = self.decode(loc.data.squeeze(0), prior_data, priors)
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # Filter the boxes less than Confidence threshold
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top_k_prediction
        order = scores.argsort()[::-1][:self.keep_top_k_pred]
        boxes = boxes[order]
        scores = scores[order]

        faces = []
        if boxes.shape[0] > 0:
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            dets = self.py_soft_nms(dets, method='greedy')
            bounding_boxes = dets[:, [0, 1, 2, 3]]
            for d in range(bounding_boxes.shape[0]):
                x_min = int(bounding_boxes[d, 0])
                y_min = int(bounding_boxes[d, 1])
                x_max = int(bounding_boxes[d, 2])
                y_max = int(bounding_boxes[d, 3])
                if frame.shape[1] >= x_max > x_min >= 0 and frame.shape[0] >= y_max > y_min >= 0:
                    faces.append([x_min, x_max, y_min, y_max])
        faces_dict = [{'area': (face[1] - face[0]) * (face[3] - face[2]), 'coordinates': face} for face in
                      faces]
        return sorted(faces_dict, key=lambda item: item['area'], reverse=True)

    def py_soft_nms(self, dets, method='linear', sigma=0.5, score_thr=0.001):
        if method not in ('linear', 'gaussian', 'greedy'):
            raise ValueError('method must be linear, gaussian or greedy')

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # expand dets with areas, and the second dimension is
        # x1, y1, x2, y2, score, area
        dets = np.concatenate((dets, areas[:, None]), axis=1)

        retained_box = []
        while dets.size > 0:
            max_idx = np.argmax(dets[:, 4], axis=0)
            dets[[0, max_idx], :] = dets[[max_idx, 0], :]
            retained_box.append(dets[0, :-1])

            xx1 = np.maximum(dets[0, 0], dets[1:, 0])
            yy1 = np.maximum(dets[0, 1], dets[1:, 1])
            xx2 = np.minimum(dets[0, 2], dets[1:, 2])
            yy2 = np.minimum(dets[0, 3], dets[1:, 3])

            w = np.maximum(xx2 - xx1 + 1, 0.0)
            h = np.maximum(yy2 - yy1 + 1, 0.0)
            inter = w * h
            iou = inter / (dets[0, 5] + dets[1:, 5] - inter)
            if method == 'linear':
                weight = np.ones_like(iou)
                weight[iou > self.iou_threshold] -= iou[iou > self.iou_threshold]
            elif method == 'gaussian':
                weight = np.exp(-(iou * iou) / sigma)
            else:  # traditional nms
                weight = np.ones_like(iou)
                weight[iou > self.iou_threshold] = 0

            dets[1:, 4] *= weight
            retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
            dets = dets[retained_idx + 1, :]

        return np.vstack(retained_box)

    def decode(self, loc, prior, priors):
        boxes_decode = torch.cat((
            prior[:, :2] + loc[:, :2] * self.variance[0] * priors[:, 2:],
            prior[:, 2:] * torch.exp(loc[:, 2:] * self.variance[1])), 1)
        boxes_decode[:, :2] -= boxes_decode[:, 2:] / 2
        boxes_decode[:, 2:] += boxes_decode[:, :2]
        return boxes_decode
