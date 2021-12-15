import os
import numpy as np

class ResultData:
    def __init__(self):
        self.bboxs = {}  # key:label , value:bbox [xmin,xmax,ymin,ymax]
        self.segmentation = None # np.ndarray (512,896,3)
        self.on_road = {} #labelがどの道の上にいるか
        self.depth = {} # key:label, value:sprCoordinates.z
# bboxの情報集め
    def collect_bb(self, label, xmin, xmax, ymin, ymax):
        self.bboxs[label] = [xmin, xmax, ymin, ymax]
# depthの情報集め
    def collect_depth(self, label, spCorz):
        self.depth[label] = spCorz
# segmentationの情報集め
    def collect_segmentaion(self,decode):
        self.segmentation = decode

    def output_bb(self):
        return self.bboxs
    
    def output_depth(self):
        return self.depth

    def is_detection(self):
        if len(self.bboxs) > 0:
            return True
        else:
            return False
# 正規化されているのを戻す
    def frameNorm(self,bbox):
        normVals = np.full(len(bbox), 512)
        normVals[::2] = 896
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# 歩道にいるか車道にいるかの検知したい
    def cal_on_road(self):
        for label,bbox in self.bboxs.items():
            bbox = self.frameNorm(bbox)
            xmin,xmax,ymin,ymax = bbox
            roadtype = self.segmentation[511-ymin,int((xmin+xmax)/2),:] 
            if roadtype[0] == 255:
                self.on_road[label] = "red"
            elif roadtype[1] == 255:
                self.on_road[label] = "green"
            elif roadtype[2] == 255:
                self.on_road[label] = "blue"
            else :
                self.on_road[label] = "none"
