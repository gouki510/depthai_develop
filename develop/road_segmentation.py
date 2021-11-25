#!/usr/bin/env python3
import blobconverter
import cv2
import depthai as dai
import numpy as np
import pickle
from depthai_sdk import PipelineManager, NNetManager, PreviewManager, Previews, FPSHandler, toTensorResult

from depthai_sdk import utils

import os

nn_shape = 896, 512

result_data = {}
def decode(packet):
    data = np.squeeze(toTensorResult(packet)["L0317_ReWeight_SoftMax"])
    class_colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    indices = np.argmax(data, axis=0)
    output_colors = np.take(class_colors, indices, axis=0)
    return output_colors


def draw(data, frame):
    if len(data) == 0:
        return
    cv2.addWeighted(frame, 1, cv2.resize(data, frame.shape[:2][::-1]), 0.2, 0, frame)


def run_all():
    # Start defining a pipeline
    pm = PipelineManager()
    pm.createColorCam(previewSize=nn_shape)

    nm = NNetManager(inputSize=nn_shape)
    pm.setNnManager(nm)
    pm.addNn(
        nm.createNN(pm.pipeline, pm.nodes, blobconverter.from_zoo(name='road-segmentation-adas-0001', shaves=6)),
        sync=True
    )
    fps = FPSHandler()
    pv = PreviewManager(display=[Previews.color.name], fpsHandler=fps)

    # Pipeline is defined, now we can connect to the device
    with dai.Device(pm.pipeline) as device:
        nm.createQueues(device)
        pv.createQueues(device)

        while True:

            fps.tick('color')
            pv.prepareFrames(blocking=True)
            frame = pv.get(Previews.color.name)

            road_decoded = decode(nm.outputQueue.get())
            draw(road_decoded, frame)
            fps.drawFps(frame, 'color')

            # pickleが空だとエラーがでるのでif分岐
            if os.path.getsize('data/data.pickle') > 0:
                # result_data の読み込み
                with open("data/data.pickle",'rb') as f:
                    result_data = pickle.load(f)
                    # 人検出ボックスの4点の座標をファイルから読み込む
                    xmin,xmax,ymin,ymax = result_data.output_bb()
                    for object_idx in range(len(xmin)):
                        print("xmax:{}, xmin:{}, ymax:{}, ymin:{}".format(xmax[object_idx], xmin[object_idx], ymax[object_idx], ymin[object_idx]))
                        #人検出ボックスの追加
                        bbox = utils.frameNorm(nm._normFrame(frame), [xmin[object_idx], ymin[object_idx], xmax[object_idx], ymax[object_idx]])
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 3)
                #フレーム完成・描画
            cv2.imshow('color', frame)

            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == "__main__":
    while True:
        if os.path.isfile("./data/rec_pos.txt"):
            break
    run_all()