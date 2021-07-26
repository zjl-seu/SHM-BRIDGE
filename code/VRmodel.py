# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:23:36 2021

@author: zjl-seu
"""
import cv2
import datetime
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from model import model_vgg16, model_rpn, model_TYPE, model_LPR

class VR:
    def __init__(self):
        #self.image = image
         
        self.chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
                       "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", 
                       "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
                       "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "港", "学", "使", "警", "澳", "挂", "军", 
                       "北", "南", "广", "沈", "兰", "成", "济", "海", "民", "航", "空"]

        self.wandhG = np.array([[80., 28.], [95., 36.], [110., 44.], [125., 52.], [140., 60.], [155., 68.], [170., 76.],
                                 [185., 84.], [200., 92.]], dtype=np.float32)
         
        self.exist_weight = "./weight/exist_model.h5"
        self.loc_weight = "./weight/loc_model.h5"
        self.lane_weight = "./weight/lane_model.h5"
        self.type_weight = "./weight/type_model.h5"
        self.lpr_weight = "./weight/lpr_model.h5"
         
        self.save_path = 'database/test/output_video/v1/'
         
        self.model_exist, self.model_loc, self.model_lane, self.model_type, self.model_lpr = self.load_model()
         
     
    def load_model(self):
        model_exist = model_vgg16()
        fake_data = np.ones(shape=[1, 128, 128, 3]).astype(np.float32)
        model_exist(fake_data)
        model_exist.load_weights(self.exist_weight)
    
        model_loc = model_rpn()
        fake_data = np.ones(shape=[1, 500, 500, 3]).astype(np.float32)
        model_loc(fake_data)
        model_loc.load_weights(self.loc_weight)
    
        model_lane = model_vgg16()
        fake_data = np.ones(shape=[1, 128, 128, 3]).astype(np.float32)
        model_lane(fake_data)
        model_lane.load_weights(self.lane_weight)
   
        model_type = model_TYPE()
        fake_data = np.ones(shape=[1, 9, 34, 3]).astype(np.float32)
        model_type(fake_data)
        model_type.load_weights(self.type_weight)
    
        model_lpr = model_LPR()
        fake_data = np.ones(shape=[1, 160, 40, 3]).astype(np.float32)
        model_lpr(fake_data)
        model_lpr.load_weights(self.lpr_weight)
        return model_exist, model_loc, model_lane, model_type, model_lpr
         
    def main(self, img):
        #img = self.image
        self.image = img
        img1 = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC) 
        img1 = (img1 / 255).astype(np.float32)
        img1 = np.expand_dims(img1, 0)
        exist_score = self.model_exist(img1)
        if exist_score.numpy() > 0.5:
            now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")   
            img2 = (img / 255).astype(np.float32)
            img2 = np.expand_dims(img2, 0)
            pred_scores, pred_bboxes = self.model_loc(img2)
            pred_scores = tf.nn.softmax(pred_scores, axis=-1)
            pred_scores, pred_bboxes = self.decode_output(pred_bboxes, pred_scores, 0.90)
            pred_bboxes, pred_scores = self.nms(pred_bboxes, pred_scores, 0.1)
            if len(pred_scores) > 0:
                plates = self.boxcrop(img, pred_bboxes)
                if len(plates) > 0:
                    lane_scores = self.model_lane(img1)
                    if lane_scores > 0.5:
                        lane = "lane2"
                    else:
                        lane = "lane1"
                    img3 = cv2.resize(plates[0][1], (34, 9), interpolation=cv2.INTER_CUBIC)
                    img3 = img3.astype(np.float32)
                    img3 = np.expand_dims(img3, 0)
                    type_scores = self.model_type(img3)
                    if type_scores > 0.5:
                        color = "blue"
                    else:
                        color = "yellow"
                    img4 = cv2.resize(plates[0][0], (160, 40), interpolation=cv2.INTER_CUBIC)
                    img4 = img4.astype(np.float32)
                    img4 = img4.transpose(1, 0, 2)
                    img4 = np.expand_dims(img4, 0)
                    lp_pred = self.model_lpr(img4)
                    lp, confidence = self.fastdecode(lp_pred[: , 2 : , :].numpy())
                    if len(lp) > 6:
                        res = now_time + "-" + lane + "-" + color + "-" + lp[:7]
                        image = self.saveimg(self.image, pred_bboxes, res)
                        return image, res    
    
    def decode_output(self, pred_bboxes, pred_scores, score_thresh=0.5):
        grid_x, grid_y = tf.range(32, dtype=tf.int32), tf.range(32, dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        grid_x, grid_y = tf.expand_dims(grid_x, -1), tf.expand_dims(grid_y, -1)
        grid_xy = tf.stack([grid_x, grid_y], axis=-1)
        center_xy = grid_xy * 16 + 8
        center_xy = tf.cast(center_xy, tf.float32)
        anchor_xymin = center_xy - 0.5 * self.wandhG

        xy_min = pred_bboxes[..., 0:2] * self.wandhG[:, 0:2] + anchor_xymin
        xy_max = tf.exp(pred_bboxes[..., 2:4]) * self.wandhG[:, 0:2] + xy_min

        pred_bboxes = tf.concat([xy_min, xy_max], axis=-1)
        pred_scores = pred_scores[..., 1]
        score_mask = pred_scores > score_thresh
        pred_bboxes = tf.reshape(pred_bboxes[score_mask], shape=[-1,4]).numpy()
        pred_scores = tf.reshape(pred_scores[score_mask], shape=[-1,]).numpy()
        return  pred_scores, pred_bboxes
    
    def nms(self, pred_boxes, pred_scores, iou_thresh):
        selected_boxes = []
        selected_scores = []
        while len(pred_boxes) > 0:
            max_idx = np.argmax(pred_scores)
            selected_box = pred_boxes[max_idx]
            selected_boxes.append(selected_box)
            selected_score = pred_scores[max_idx]
            selected_scores.append(selected_score)
            pred_boxes = np.concatenate([pred_boxes[:max_idx], pred_boxes[max_idx+1:]])
            pred_scores = np.concatenate([pred_scores[:max_idx], pred_scores[max_idx+1:]])
            ious = self.compute_iou(selected_box, pred_boxes)
            iou_mask = ious <= iou_thresh
            pred_boxes = pred_boxes[iou_mask]
            pred_scores = pred_scores[iou_mask]

        selected_boxes = np.array(selected_boxes)
        selected_scores = np.array(selected_scores)
        return selected_boxes, selected_scores
    
    def compute_iou(self, boxes1, boxes2):
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2], )
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_wh = np.maximum(right_down - left_up, 0.0)  # 交集的宽和长
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]  # 交集的面积

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  

        union_area = boxes1_area + boxes2_area - inter_area  
        ious = inter_area / union_area
        return ious
    
    def boxcrop(self, image, boxes):
         cropped_images = [] 
         for box in boxes:
             drop = self.bounding_detect(image, box)
             if drop == True:
                 continue
             cropped_origin = image[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
             w = box[2]-box[0]
             h = box[3]-box[1]
             top = max(int(box[1] - 0.1 * h),0)
             bottom = min(int(box[3] + 0.1 * h), image.shape[0])
             left = max(int(box[0] - 0.15 * w), 0)
             right = min(int(box[2] + 0.15 * w), image.shape[1])
             cropped = image[top : bottom, left : right]
             cropped_images.append([cropped, cropped_origin])
         return cropped_images
     
    def bounding_detect(self, image,bounding_rect):
        shape = image.shape
        top = bounding_rect[1] 
        bottom = bounding_rect[3] 
        left = bounding_rect[0] 
        right = bounding_rect[2] 

        min_top = 0
        max_bottom = shape[0]
        min_left = 0
        max_right = shape[1]

        if (top > min_top) and (left > min_left) and (bottom < max_bottom) and (right < max_right): 
            drop = False
        else:
            drop = True
        return drop

    def fastdecode(self, y_pred):
        results = ""
        confidence = 0.0
        table_pred = y_pred.reshape(-1, len(self.chars) + 1)
        res = table_pred.argmax(axis = 1)
        while True:
            if res[0] > 30 and res[0] < 65:
                res = res[1:]
            else:
                break
        for i, one in enumerate(res):
            if one < len(self.chars) and (i == 0 or (one != res[i - 1])):
                results += self.chars[one]
                confidence += table_pred[i][one]
        if len(results) != 0: 
            confidence /= len(results)
            return results, confidence
        else:
            return "0", "0" 

    def saveimg(self, image, boxes=[], addText=""):
        fontc = ImageFont.truetype("simsun.ttc", 40, encoding="unic")
        cv2.rectangle(image, (int(800), int(300)), (int(1300), int(800)), (0, 0, 255), 2, cv2.LINE_AA)
        if addText != "":
            cv2.rectangle(image, (int(boxes[0, 0] + 800 - 0.15*(boxes[0, 2] - boxes[0, 0])), int(boxes[0, 1] + 300)),
                                 (int(boxes[0, 2] + 800 + 0.15 * (boxes[0, 2] - boxes[0, 0])), int(boxes[0, 3] + 300)), (0, 0, 255), 2, cv2.LINE_AA)
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            draw.text((int(boxes[0, 0] - 120 + 800), int(boxes[0, 1] - 40 + 300)), addText, (0, 0, 255), font=fontc)
        imagex = np.array(image) 
        return imagex

if __name__ == "__main__":
    image = cv2.imread("camera 00262.jpg")
    vr = VR()
    image, res = vr.main(image)
    print(res)
    

    
    
    
