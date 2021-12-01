#!/usr/bin/env python3
import blobconverter
import cv2
import depthai as dai
import numpy as np
import pickle
from depthai_sdk import (
    PipelineManager,
    NNetManager,
    PreviewManager,
    Previews,
    FPSHandler,
    toTensorResult,
)

from depthai_sdk import utils

import os

from numpy.core.records import array

from PIL import Image

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
        nm.createNN(
            pm.pipeline,
            pm.nodes,
            blobconverter.from_zoo(name="road-segmentation-adas-0001", shaves=6),
        ),
        sync=True,
    )
    fps = FPSHandler()
    pv = PreviewManager(display=[Previews.color.name], fpsHandler=fps)

    # Pipeline is defined, now we can connect to the device
    with dai.Device(pm.pipeline) as device:
        nm.createQueues(device)
        pv.createQueues(device)

        while True:

            fps.tick("color")
            pv.prepareFrames(blocking=True)
            frame = pv.get(Previews.color.name)

            road_decoded = decode(nm.outputQueue.get())
            # ロード・セグメンテーションの結果を画像に保存
            im = Image.fromarray(road_decoded)
            im.save("result_roadseg.png")

            draw(road_decoded, frame)
            fps.drawFps(frame, "color")

            # pickleが空だとエラーがでるのでif分岐
            if os.path.getsize("data/data.pickle") > 0:
                # result_data の読み込み
                with open("data/data.pickle", "rb") as f:
                    result_data = pickle.load(f)
                    # 人検出ボックスの4点の座標をファイルから読み込む
                    output_dic = result_data.output_bb()
                    # segmentation 情報をresultdataに格納
                    result_data.collect_segmentaion(road_decoded)
                    for label in output_dic.keys():
                        xmin, xmax, ymin, ymax = (
                            output_dic[label][0],
                            output_dic[label][1],
                            output_dic[label][2],
                            output_dic[label][3],
                        )
                        # 人検出ボックスの追加
                        bbox = utils.frameNorm(
                            nm._normFrame(frame), [xmin, ymin, xmax, ymax]
                        )
                        result_data.cal_on_road()
                        print("label:{},xmin:{}, ymin:{}, xmax:{}, ymax:{}".format(label,bbox[0],bbox[1],bbox[2],bbox[3]))
                        cv2.rectangle(
                            frame,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            (0, 255, 0),
                            3,
                        )
                # フレーム完成・描画
            cv2.imshow("color", frame)

            if cv2.waitKey(1) == ord("q"):
                break


if __name__ == "__main__":
    run_all()

