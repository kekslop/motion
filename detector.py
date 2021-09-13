import numpy as np
import pyrealsense2 as rs
import os.path

from time import time
from detector import MotionDetector
from packer import pack_images
from numba import jit


@jit(nopython=True)
def filter_fun(b):
    return ((b[2] - b[0]) * (b[3] - b[1])) > 300


if __name__ == "__main__":

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config,"file.bag")
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
    prof = pipeline.start(config)
    
    #when read from sensor
    #s = prof.get_device().query_sensors()[1]
    #s.set_option(rs.option.exposure, 450)
    
    colorizer = rs.colorizer();
    cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
    	
    detector = MotionDetector(bg_history=10,
                              bg_skip_frames=5,
                              movement_frames_history=5,
                              brightness_discard_level=95,
                              bg_subs_scale_percent=0.5,
                              pixel_compression_ratio=0.5,
                              group_boxes=True,
                              expansion_step=5)

    # group_boxes=True can be used if one wants to get less boxes, which include all overlapping boxes

    b_height = 640
    b_width = 480
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_frame = np.asanyarray(color_frame.get_data())
    #config.enable_record_to_file('arnest22.bag')
    r = cv2.selectROI(color_frame,False, False)
    print(r)
    res = []
    fc = dict()
    ctr = 0
    while True:
        # Capture frame-by-frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_frame = np.asanyarray(color_frame.get_data())
        frame = color_frame
        frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        #print(frame)
        if frame is None:
            break

        begin = time()

        boxes, color_frame = detector.detect(frame)
        # boxes hold all boxes around motion parts

        ## this code cuts motion areas from initial image and
        ## fills "bins" of 320x320 with such motion areas.
        ##
        results = []
        if boxes:
             results, box_map = pack_images(frame=frame, boxes=boxes, width=b_width, height=b_height,
                                            box_filter=filter_fun)
            # box_map holds list of mapping between image placement in packed bins and original boxes

        ## end

        for b in boxes:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)

        end = time()
        it = (end - begin) * 1000

        res.append(it)
        print("StdDev: %.4f" % np.std(res), "Mean: %.4f" % np.mean(res), "Last: %.4f" % it,
              "Boxes found: ", len(boxes))

        if len(res) > 10000:
            res = []

        # idx = 0
        # for r in results:
        #      idx += 1
        #      cv2.imshow('packed_frame_%d' % idx, r)

        ctr += 1
        nc = len(results)
        if nc in fc:
            fc[nc] += 1
        else:
            fc[nc] = 0

        if ctr % 100 == 0:
            print("Total Frames: ", ctr, "Packed Frames:", fc)

        
        cv2.imshow("Color Stream", frame)
        cv2.imshow('detect_frame', detector.detection_boxed)
        cv2.imshow('diff_frame', detector.color_movement)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(fc, ctr)
