#!/usr/bin/env python3
import os
from itertools import cycle
from pathlib import Path
from subprocess import run
import cv2
import depthai as dai
import platform
import numpy as np
from myutils import ResultData
import pickle

#road_segmentation.py imports
import blobconverter
from depthai_sdk import utils


from depthai_helpers.arg_manager import parseArgs
from depthai_helpers.config_manager import ConfigManager, DEPTHAI_ZOO, DEPTHAI_VIDEOS
from depthai_helpers.version_check import checkRequirementsVersion
from depthai_sdk import FPSHandler, loadModule, getDeviceInfo, downloadYTVideo, Previews
from depthai_sdk import (
    NNetManager,
    PreviewManager,
    PipelineManager,
    EncodingManager,
    BlobManager,

    toTensorResult,
    Previews,

)


DISP_CONF_MIN = int(os.getenv("DISP_CONF_MIN", 0))
DISP_CONF_MAX = int(os.getenv("DISP_CONF_MAX", 255))
SIGMA_MIN = int(os.getenv("SIGMA_MIN", 0))
SIGMA_MAX = int(os.getenv("SIGMA_MAX", 250))
LRCT_MIN = int(os.getenv("LRCT_MIN", 0))
LRCT_MAX = int(os.getenv("LRCT_MAX", 10))


"""road_segmentation func insert start"""
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
"""road_segmentation func insert end"""



def run_all():

    xmax, xmin, ymax, ymin = 0, 0, 0, 0
    print("Using depthai module from: ", dai.__file__)
    print("Depthai version installed: ", dai.__version__)
    if platform.machine() not in ["armv6l", "aarch64"]:
        checkRequirementsVersion()

    conf = ConfigManager(parseArgs())
    conf.linuxCheckApplyUsbRules()
    if not conf.useCamera:
        pass

    callbacks = loadModule(conf.args.callback)
    rgbRes = conf.getRgbResolution()
    monoRes = conf.getMonoResolution()

    if conf.args.reportFile:
        reportFileP = Path(conf.args.reportFile).with_suffix(".csv")
        reportFileP.parent.mkdir(parents=True, exist_ok=True)
        reportFile = open(conf.args.reportFile, "a")

    def printSysInfo(info):
        m = 1024 * 1024  # MiB
        if not conf.args.reportFile:
            if "memory" in conf.args.report:
                print(
                    f"Drr used / total - {info.ddrMemoryUsage.used / m:.2f} / {info.ddrMemoryUsage.total / m:.2f} MiB"
                )
                print(
                    f"Cmx used / total - {info.cmxMemoryUsage.used / m:.2f} / {info.cmxMemoryUsage.total / m:.2f} MiB"
                )
                print(
                    f"LeonCss heap used / total - {info.leonCssMemoryUsage.used / m:.2f} / {info.leonCssMemoryUsage.total / m:.2f} MiB"
                )
                print(
                    f"LeonMss heap used / total - {info.leonMssMemoryUsage.used / m:.2f} / {info.leonMssMemoryUsage.total / m:.2f} MiB"
                )
            if "temp" in conf.args.report:
                t = info.chipTemperature
                print(
                    f"Chip temperature - average: {t.average:.2f}, css: {t.css:.2f}, mss: {t.mss:.2f}, upa0: {t.upa:.2f}, upa1: {t.dss:.2f}"
                )
            if "cpu" in conf.args.report:
                print(
                    f"Cpu usage - Leon OS: {info.leonCssCpuUsage.average * 100:.2f}%, Leon RT: {info.leonMssCpuUsage.average * 100:.2f} %"
                )
            print("----------------------------------------")
        else:
            data = {}
            if "memory" in conf.args.report:
                data = {
                    **data,
                    "ddrUsed": info.ddrMemoryUsage.used,
                    "ddrTotal": info.ddrMemoryUsage.total,
                    "cmxUsed": info.cmxMemoryUsage.used,
                    "cmxTotal": info.cmxMemoryUsage.total,
                    "leonCssUsed": info.leonCssMemoryUsage.used,
                    "leonCssTotal": info.leonCssMemoryUsage.total,
                    "leonMssUsed": info.leonMssMemoryUsage.used,
                    "leonMssTotal": info.leonMssMemoryUsage.total,
                }
            if "temp" in conf.args.report:
                data = {
                    **data,
                    "tempAvg": info.chipTemperature.average,
                    "tempCss": info.chipTemperature.css,
                    "tempMss": info.chipTemperature.mss,
                    "tempUpa0": info.chipTemperature.upa,
                    "tempUpa1": info.chipTemperature.dss,
                }
            if "cpu" in conf.args.report:
                data = {
                    **data,
                    "cpuCssAvg": info.leonCssCpuUsage.average,
                    "cpuMssAvg": info.leonMssCpuUsage.average,
                }

            if reportFile.tell() == 0:
                print(",".join(data.keys()), file=reportFile)
            callbacks.onReport(data)
            print(",".join(map(str, data.values())), file=reportFile)

    # いる？
    class Trackbars:
        instances = {}

        @staticmethod
        def createTrackbar(name, window, minVal, maxVal, defaultVal, callback):
            def fn(value):
                if Trackbars.instances[name][window] != value:
                    callback(value)
                for otherWindow, previousValue in Trackbars.instances[name].items():
                    if otherWindow != window and previousValue != value:
                        Trackbars.instances[name][otherWindow] = value
                        cv2.setTrackbarPos(name, otherWindow, value)

            cv2.createTrackbar(name, window, minVal, maxVal, fn)
            Trackbars.instances[name] = {
                **Trackbars.instances.get(name, {}),
                window: defaultVal,
            }
            cv2.setTrackbarPos(name, window, defaultVal)

    deviceInfo = getDeviceInfo(conf.args.deviceId)
    openvinoVersion = None
    if conf.args.openvinoVersion:
        openvinoVersion = getattr(
            dai.OpenVINO.Version, "VERSION_" + conf.args.openvinoVersion
        )
    pm = PipelineManager(openvinoVersion)

    if conf.args.xlinkChunkSize is not None:
        pm.setXlinkChunkSize(conf.args.xlinkChunkSize)

    if conf.useNN:
        blobManager = BlobManager(zooDir=DEPTHAI_ZOO, zooName=conf.getModelName(),)
        nnManager = NNetManager(inputSize=conf.inputSize)

        if conf.getModelDir() is not None:
            configPath = conf.getModelDir() / Path(conf.getModelName()).with_suffix(
                f".json"
            )
            nnManager.readConfig(configPath)

        nnManager.countLabel(conf.getCountLabel(nnManager))
        pm.setNnManager(nnManager)

    """road_segmentation insert start"""
    # Start defining a pipeline
    nn_shape = 896, 512
    result_data1 = {}
    pm1 = PipelineManager()
    pm1.createColorCam(previewSize=nn_shape)

    nm1 = NNetManager(inputSize=nn_shape)
    pm1.setNnManager(nm1)
    pm1.addNn(
        nm1.createNN(
            pm1.pipeline,
            pm1.nodes,
            blobconverter.from_zoo(name="road-segmentation-adas-0001", shaves=6),
        ),
        sync=True,
    )
    fps1 = FPSHandler()
    pv1 = PreviewManager(display=[Previews.color.name], fpsHandler=fps1)
    """road_segmentation insert end"""

    """road_segmentation dai insert start"""
    # Pipeline is defined, now we can connect to the device
    with dai.Device(pm1.pipeline) as device1:
        nm1.createQueues(device1)
        pv1.createQueues(device1)
        print("road_seg_dai")
        if deviceInfo.desc.protocol == dai.XLinkProtocol.X_LINK_USB_VSC:
            print("USB Connection speed: {}".format(device1.getUsbSpeed()))
        """road_segmentation dai insert end"""


        # Pipeline is defined, now we can connect to the device
        with dai.Device(
            pm.pipeline.getOpenVINOVersion(),
            deviceInfo,
            #usb2Mode=conf.args.usbSpeed == "usb2",
        ) as device:
            print("demo_dai")
            if deviceInfo.desc.protocol == dai.XLinkProtocol.X_LINK_USB_VSC:
                print("USB Connection speed: {}".format(device.getUsbSpeed()))
            conf.adjustParamsToDevice(device)
            conf.adjustPreviewToOptions()
            if conf.lowBandwidth:
                pm.enableLowBandwidth()
            cap = cv2.VideoCapture(conf.args.video) if not conf.useCamera else None
            fps = FPSHandler() if conf.useCamera else FPSHandler(cap)

            if conf.useCamera or conf.args.sync:
                pv = PreviewManager(
                    display=conf.args.show,
                    nnSource=conf.getModelSource(),
                    colorMap=conf.getColorMap(),
                    dispMultiplier=conf.dispMultiplier,
                    mouseTracker=True,
                    lowBandwidth=conf.lowBandwidth,
                    scale=conf.args.scale,
                    sync=conf.args.sync,
                    fpsHandler=fps,
                )

                if conf.leftCameraEnabled:
                    pm.createLeftCam(
                        monoRes,
                        conf.args.monoFps,
                        orientation=conf.args.cameraOrientation.get(Previews.left.name),
                        xout=Previews.left.name in conf.args.show
                        and (conf.getModelSource() != "left" or not conf.args.sync),
                    )
                if conf.rightCameraEnabled:
                    pm.createRightCam(
                        monoRes,
                        conf.args.monoFps,
                        orientation=conf.args.cameraOrientation.get(Previews.right.name),
                        xout=Previews.right.name in conf.args.show
                        and (conf.getModelSource() != "right" or not conf.args.sync),
                    )
                if conf.rgbCameraEnabled:
                    pm.createColorCam(
                        nnManager.inputSize if conf.useNN else conf.previewSize,
                        rgbRes,
                        conf.args.rgbFps,
                        orientation=conf.args.cameraOrientation.get(Previews.color.name),
                        fullFov=not conf.args.disableFullFovNn,
                        xout=Previews.color.name in conf.args.show
                        and (conf.getModelSource() != "color" or not conf.args.sync),
                    )

                if conf.useDepth:
                    pm.createDepth(
                        conf.args.disparityConfidenceThreshold,
                        conf.getMedianFilter(),
                        conf.args.sigma,
                        conf.args.stereoLrCheck,
                        conf.args.lrcThreshold,
                        conf.args.extendedDisparity,
                        conf.args.subpixel,
                        useDepth=Previews.depth.name in conf.args.show
                        or Previews.depthRaw.name in conf.args.show,
                        useDisparity=Previews.disparity.name in conf.args.show
                        or Previews.disparityColor.name in conf.args.show,
                        useRectifiedLeft=Previews.rectifiedLeft.name in conf.args.show
                        and (
                            conf.getModelSource() != "rectifiedLeft" or not conf.args.sync
                        ),
                        useRectifiedRight=Previews.rectifiedRight.name in conf.args.show
                        and (
                            conf.getModelSource() != "rectifiedRight" or not conf.args.sync
                        ),
                    )

                encManager = None
                if len(conf.args.encode) > 1:
                    encManager = EncodingManager(conf.args.encode, conf.args.encodeOutput)
                    encManager.createEncoders(pm)

            if len(conf.args.report) > 0:
                pm.createSystemLogger()

            if conf.useNN:
                nn = nnManager.createNN(
                    pipeline=pm.pipeline,
                    nodes=pm.nodes,
                    source=conf.getModelSource(),
                    blobPath=blobManager.getBlob(
                        shaves=conf.shaves, openvinoVersion=nnManager.openvinoVersion
                    ),
                    useDepth=conf.useDepth,
                    minDepth=conf.args.minDepth,
                    maxDepth=conf.args.maxDepth,
                    sbbScaleFactor=conf.args.sbbScaleFactor,
                    fullFov=not conf.args.disableFullFovNn,
                    flipDetection=conf.getModelSource()
                    in ("rectifiedLeft", "rectifiedRight")
                    and not conf.args.stereoLrCheck,
                )

                pm.addNn(
                    nn=nn,
                    sync=conf.args.sync,
                    xoutNnInput=Previews.nnInput.name in conf.args.show,
                    useDepth=conf.useDepth,
                    xoutSbb=conf.args.spatialBoundingBox and conf.useDepth,
                )

            # Start pipeline
            device.startPipeline(pm.pipeline)
            pm.createDefaultQueues(device)
            if conf.useNN:
                nnManager.createQueues(device)

            sbbOut = (
                device.getOutputQueue("sbb", maxSize=1, blocking=False)
                if conf.useNN and conf.args.spatialBoundingBox
                else None
            )
            logOut = (
                device.getOutputQueue("systemLogger", maxSize=30, blocking=False)
                if len(conf.args.report) > 0
                else None
            )

            medianFilters = cycle(
                [
                    item
                    for name, item in vars(dai.MedianFilter).items()
                    if name.startswith("KERNEL_") or name.startswith("MEDIAN_")
                ]
            )
            for medFilter in medianFilters:
                # move the cycle to the current median filter
                if medFilter == pm._depthConfig.getMedianFilter():
                    break

            if conf.useCamera:

                def createQueueCallback(queueName):
                    if queueName in [
                        Previews.disparityColor.name,
                        Previews.disparity.name,
                        Previews.depth.name,
                        Previews.depthRaw.name,
                    ]:
                        Trackbars.createTrackbar(
                            "Disparity confidence",
                            queueName,
                            DISP_CONF_MIN,
                            DISP_CONF_MAX,
                            conf.args.disparityConfidenceThreshold,
                            lambda value: pm.updateDepthConfig(device, dct=value),
                        )
                        if queueName in [Previews.depthRaw.name, Previews.depth.name]:
                            Trackbars.createTrackbar(
                                "Bilateral sigma",
                                queueName,
                                SIGMA_MIN,
                                SIGMA_MAX,
                                conf.args.sigma,
                                lambda value: pm.updateDepthConfig(device, sigma=value),
                            )
                        if conf.args.stereoLrCheck:
                            Trackbars.createTrackbar(
                                "LR-check threshold",
                                queueName,
                                LRCT_MIN,
                                LRCT_MAX,
                                conf.args.lrcThreshold,
                                lambda value: pm.updateDepthConfig(
                                    device, lrcThreshold=value
                                ),
                            )

                cameras = device.getConnectedCameras()
                if (
                    dai.CameraBoardSocket.LEFT in cameras
                    and dai.CameraBoardSocket.RIGHT in cameras
                ):
                    pv.collectCalibData(device)

                cameraConfig = {
                    "exposure": conf.args.cameraExposure,
                    "sensitivity": conf.args.cameraSensitivity,
                    "saturation": conf.args.cameraSaturation,
                    "contrast": conf.args.cameraContrast,
                    "brightness": conf.args.cameraBrightness,
                    "sharpness": conf.args.cameraSharpness,
                }

                def updateCameraConfigs():
                    if conf.leftCameraEnabled:
                        pm.updateLeftCamConfig(device, **cameraConfig)
                    if conf.rightCameraEnabled:
                        pm.updateRightCamConfig(device, **cameraConfig)
                    if conf.rgbCameraEnabled:
                        pm.updateColorCamConfig(device, **cameraConfig)

                if any(cameraConfig.values()):
                    updateCameraConfigs()

                pv.createQueues(device, createQueueCallback)
                if encManager is not None:
                    encManager.createDefaultQueues(device)
            elif conf.args.sync:
                hostOut = device.getOutputQueue(
                    Previews.nnInput.name, maxSize=1, blocking=False
                )

            seqNum = 0
            hostFrame = None
            nnData = []
            sbbRois = []
            callbacks.onSetup(**locals())

            try:
                while True:

                    fps.nextIter()
                    callbacks.onIter(**locals())
                    if conf.useCamera:
                        pv.prepareFrames(callback=callbacks.onNewFrame)
                        if encManager is not None:
                            encManager.parseQueues()

                        if sbbOut is not None:
                            sbb = sbbOut.tryGet()
                            if sbb is not None:
                                sbbRois = sbb.getConfigData()
                            depthFrames = [
                                pv.get(Previews.depthRaw.name),
                                pv.get(Previews.depth.name),
                            ]
                            for depthFrame in depthFrames:
                                if depthFrame is None:
                                    continue

                                for roiData in sbbRois:
                                    roi = roiData.roi.denormalize(
                                        depthFrame.shape[1], depthFrame.shape[0]
                                    )
                                    topLeft = roi.topLeft()
                                    bottomRight = roi.bottomRight()
                                    # Display SBB on the disparity map
                                    cv2.rectangle(
                                        depthFrame,
                                        (int(topLeft.x), int(topLeft.y)),
                                        (int(bottomRight.x), int(bottomRight.y)),
                                        nnManager._bboxColors[0],
                                        2,
                                    )
                    else:
                        readCorrectly, rawHostFrame = cap.read()
                        if not readCorrectly:
                            break

                        nnManager.sendInputFrame(rawHostFrame, seqNum)
                        seqNum += 1

                        if not conf.args.sync:
                            hostFrame = rawHostFrame
                            print(type(hostFrame))
                        fps.tick("host")

                    if conf.useNN:
                        inNn = nnManager.outputQueue.tryGet()
                        if inNn is not None:
                            callbacks.onNn(inNn)
                            if not conf.useCamera and conf.args.sync:
                                hostFrame = Previews.nnInput.value(hostOut.get())
                            nnData = nnManager.decode(inNn)
                            fps.tick("nn")

                    if len(nnData) > 0:
                        """ print("label Text : {}".format(nnManager.getLabelText(nnData[0].label)))
                        label_Text = nnManager.getLabelText(nnData[0].label)
                        label = nnData[0].label
                        confidence = nnData[0].confidence  """
                        """ spCor_x = nnData[0].spatialCoordinates.x
                        spCor_y = nnData[0].spatialCoordinates.y
                        spCor_z = nnData[0].spatialCoordinates.z """
                        # 人物のbboxをpickleにして保存
                        # data dir をつくっておく
                        os.makedirs("data", exist_ok=True)
                        with open("data/data.pickle", "wb") as f:
                            result_data = ResultData()
                            # objectの数だけresult_dataに保存
                            for object_idx in range(len(nnData)):
                                label = nnManager.getLabelText(nnData[object_idx].label)
                                # personが複数人になったときはperson+object_idx
                                if label in result_data.bboxs.keys():
                                    label = label + str(object_idx)
                                xmax = nnData[object_idx].xmax
                                xmin = nnData[object_idx].xmin
                                ymax = nnData[object_idx].ymax
                                ymin = nnData[object_idx].ymin
                                result_data.collect_bb(label, xmin, xmax, ymin, ymax)
                                print(result_data.output_bb())
                            pickle.dump(result_data, f)

                    if conf.useCamera:
                        if conf.useNN:
                            nnManager.draw(pv, nnData)

                        def showFramesCallback(frame, name):
                            fps.drawFps(frame, name)
                            h, w = frame.shape[:2]
                            if name in [
                                Previews.disparityColor.name,
                                Previews.disparity.name,
                                Previews.depth.name,
                                Previews.depthRaw.name,
                            ]:
                                text = "Median filter: {} [M]".format(
                                    pm._depthConfig.getMedianFilter()
                                    .name.lstrip("KERNEL_")
                                    .lstrip("MEDIAN_")
                                )
                                cv2.putText(
                                    frame,
                                    text,
                                    (10, h - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    0,
                                    4,
                                )
                                cv2.putText(
                                    frame,
                                    text,
                                    (10, h - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    255,
                                    1,
                                )
                            elif conf.args.cameraControlls and name in [
                                Previews.color.name,
                                Previews.left.name,
                                Previews.right.name,
                            ]:
                                text = "Exposure: {}   T [+] [-] G".format(
                                    cameraConfig["exposure"]
                                    if cameraConfig["exposure"] is not None
                                    else "auto"
                                )
                                label_width = cv2.getTextSize(
                                    text, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 4
                                )[0][0]
                                cv2.putText(
                                    frame,
                                    text,
                                    (w - label_width, h - 110),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    4,
                                )
                                cv2.putText(
                                    frame,
                                    text,
                                    (w - label_width, h - 110),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                )
                                text = "Sensitivity: {}   Y [+] [-] H".format(
                                    cameraConfig["sensitivity"]
                                    if cameraConfig["sensitivity"] is not None
                                    else "auto"
                                )
                                label_width = cv2.getTextSize(
                                    text, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 4
                                )[0][0]
                                cv2.putText(
                                    frame,
                                    text,
                                    (w - label_width, h - 90),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    4,
                                )
                                cv2.putText(
                                    frame,
                                    text,
                                    (w - label_width, h - 90),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                )
                                text = "Saturation: {}   U [+] [-] J".format(
                                    cameraConfig["saturation"]
                                    if cameraConfig["saturation"] is not None
                                    else "auto"
                                )
                                label_width = cv2.getTextSize(
                                    text, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 4
                                )[0][0]
                                cv2.putText(
                                    frame,
                                    text,
                                    (w - label_width, h - 70),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    4,
                                )
                                cv2.putText(
                                    frame,
                                    text,
                                    (w - label_width, h - 70),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                )
                                text = "Contrast: {}   I [+] [-] K".format(
                                    cameraConfig["contrast"]
                                    if cameraConfig["contrast"] is not None
                                    else "auto"
                                )
                                label_width = cv2.getTextSize(
                                    text, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 4
                                )[0][0]
                                cv2.putText(
                                    frame,
                                    text,
                                    (w - label_width, h - 50),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    4,
                                )
                                cv2.putText(
                                    frame,
                                    text,
                                    (w - label_width, h - 50),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                )
                                text = "Brightness: {}   O [+] [-] L".format(
                                    cameraConfig["brightness"]
                                    if cameraConfig["brightness"] is not None
                                    else "auto"
                                )
                                label_width = cv2.getTextSize(
                                    text, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 4
                                )[0][0]
                                cv2.putText(
                                    frame,
                                    text,
                                    (w - label_width, h - 30),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    4,
                                )
                                cv2.putText(
                                    frame,
                                    text,
                                    (w - label_width, h - 30),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                )
                                text = "Sharpness: {}   P [+] [-] ;".format(
                                    cameraConfig["sharpness"]
                                    if cameraConfig["sharpness"] is not None
                                    else "auto"
                                )
                                label_width = cv2.getTextSize(
                                    text, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 4
                                )[0][0]
                                cv2.putText(
                                    frame,
                                    text,
                                    (w - label_width, h - 10),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    4,
                                )
                                cv2.putText(
                                    frame,
                                    text,
                                    (w - label_width, h - 10),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                )
                            returnFrame = callbacks.onShowFrame(frame, name)
                            return returnFrame if returnFrame is not None else frame

                        pv.showFrames(callback=showFramesCallback)
                    elif hostFrame is not None:
                        debugHostFrame = hostFrame.copy()
                        if conf.useNN:
                            nnManager.draw(debugHostFrame, nnData)
                        fps.drawFps(debugHostFrame, "host")
                        cv2.imshow("host", debugHostFrame)

                    if logOut:
                        logs = logOut.tryGetAll()
                        for log in logs:
                            printSysInfo(log)

                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break
                    elif key == ord("m"):
                        nextFilter = next(medianFilters)
                        pm.updateDepthConfig(device, median=nextFilter)

                    if conf.args.cameraControlls:
                        update = True

                        if key == ord("t"):
                            cameraConfig["exposure"] = (
                                10000
                                if cameraConfig["exposure"] is None
                                else 500
                                if cameraConfig["exposure"] == 1
                                else min(cameraConfig["exposure"] + 500, 33000)
                            )
                            if cameraConfig["sensitivity"] is None:
                                cameraConfig["sensitivity"] = 800
                        elif key == ord("g"):
                            cameraConfig["exposure"] = (
                                10000
                                if cameraConfig["exposure"] is None
                                else max(cameraConfig["exposure"] - 500, 1)
                            )
                            if cameraConfig["sensitivity"] is None:
                                cameraConfig["sensitivity"] = 800
                        elif key == ord("y"):
                            cameraConfig["sensitivity"] = (
                                800
                                if cameraConfig["sensitivity"] is None
                                else min(cameraConfig["sensitivity"] + 50, 1600)
                            )
                            if cameraConfig["exposure"] is None:
                                cameraConfig["exposure"] = 10000
                        elif key == ord("h"):
                            cameraConfig["sensitivity"] = (
                                800
                                if cameraConfig["sensitivity"] is None
                                else max(cameraConfig["sensitivity"] - 50, 100)
                            )
                            if cameraConfig["exposure"] is None:
                                cameraConfig["exposure"] = 10000
                        elif key == ord("u"):
                            cameraConfig["saturation"] = (
                                0
                                if cameraConfig["saturation"] is None
                                else min(cameraConfig["saturation"] + 1, 10)
                            )
                        elif key == ord("j"):
                            cameraConfig["saturation"] = (
                                0
                                if cameraConfig["saturation"] is None
                                else max(cameraConfig["saturation"] - 1, -10)
                            )
                        elif key == ord("i"):
                            cameraConfig["contrast"] = (
                                0
                                if cameraConfig["contrast"] is None
                                else min(cameraConfig["contrast"] + 1, 10)
                            )
                        elif key == ord("k"):
                            cameraConfig["contrast"] = (
                                0
                                if cameraConfig["contrast"] is None
                                else max(cameraConfig["contrast"] - 1, -10)
                            )
                        elif key == ord("o"):
                            cameraConfig["brightness"] = (
                                0
                                if cameraConfig["brightness"] is None
                                else min(cameraConfig["brightness"] + 1, 10)
                            )
                        elif key == ord("l"):
                            cameraConfig["brightness"] = (
                                0
                                if cameraConfig["brightness"] is None
                                else max(cameraConfig["brightness"] - 1, -10)
                            )
                        elif key == ord("p"):
                            cameraConfig["sharpness"] = (
                                0
                                if cameraConfig["sharpness"] is None
                                else min(cameraConfig["sharpness"] + 1, 4)
                            )
                        elif key == ord(";"):
                            cameraConfig["sharpness"] = (
                                0
                                if cameraConfig["sharpness"] is None
                                else max(cameraConfig["sharpness"] - 1, 0)
                            )
                        else:
                            update = False

                        if update:
                            updateCameraConfigs()


                        """road_segmentation insert start"""

                        fps1.tick("color")
                        pv1.prepareFrames(blocking=True)
                        frame1 = pv1.get(Previews.color.name)

                        road_decoded = decode(nm1.outputQueue.get())
                        draw(road_decoded, frame1)
                        fps1.drawFps(frame1, "color")

                        # pickleが空だとエラーがでるのでif分岐
                        if os.path.getsize("data/data.pickle") > 0:
                            # result_data の読み込み
                            with open("data/data.pickle", "rb") as f:
                                result_data1 = pickle.load(f)
                                # 人検出ボックスの4点の座標をファイルから読み込む
                                output_dic = result_data1.output_bb()
                                # segmentation 情報をresultdataに格納
                                result_data1.collect_segmentaion(road_decoded)
                                for label in output_dic.keys():
                                    xmin1, xmax1, ymin1, ymax1 = (
                                        output_dic[label][0],
                                        output_dic[label][1],
                                        output_dic[label][2],
                                        output_dic[label][3],
                                    )
                                    # 人検出ボックスの追加
                                    bbox1 = utils.frameNorm(
                                    nm1._normFrame(frame1), [xmin1, ymin1, xmax1, ymax1]
                                    )
                                    result_data1.cal_on_road()
                                    print("label:{},xmin:{}, ymin:{}, xmax:{}, ymax:{}".format(label,bbox1[0],bbox1[1],bbox1[2],bbox1[3]))
                                    cv2.rectangle(
                                        frame1,
                                        (bbox1[0], bbox1[1]),
                                        (bbox1[2], bbox1[3]),
                                        (0, 255, 0),
                                        3,
                                    )
                        # フレーム完成・描画
                        cv2.imshow("color", frame1)

            finally:
                if conf.useCamera and encManager is not None:
                    encManager.close()

    if conf.args.reportFile:
        reportFile.close()

    fps.printStatus()
    callbacks.onTeardown(**locals())


if __name__ == "__main__":
    run_all()
