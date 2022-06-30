import argparse
import numpy as np
import cv2
import os
import sys


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("cptv", help="CPTV file to parse")

    args = parser.parse_args()
    return args


def detect_objects(image, otsus=True, threshold=0, kernel=(15, 15)):

    # cv2.imshow("Ca", image)
    # cv2.waitKey(0)
    # image -= np.mean(image)
    # image = cv2.fastNlMeansDenoising(np.uint8(image), None)

    # image = cv2.GaussianBlur(image, kernel, 0)
    # cv2.imshow("gauss", image)
    # cv2.waitKey(0)
    # sobelxy = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1)  #
    image = cv2.Canny(image, 150, 200)
    # return

    # flags = cv2.THRESH_BINARY
    # if otsus:
    #     flags += cv2.THRESH_OTSU
    # _, image = cv2.threshold(image, threshold, 255, flags)
    # cv2.imshow("otsus", image)

    # cv2.waitKey(0)
    image = cv2.dilate(image, kernel, iterations=1)
    # cv2.imshow("dilate", image)
    #
    # cv2.waitKey(0)
    #
    # flags = cv2.THRESH_BINARY
    # if otsus:
    #     flags += cv2.THRESH_OTSU
    # _, image = cv2.threshold(image, threshold, 255, flags)
    # print("OTSUSSSS")

    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("otsus", image)

    # cv2.imshow("morphologyEx", image)
    # cv2.waitKey(0)

    # return image
    # import matplotlib.pyplot as plt
    #
    # imgplot = plt.imshow(image)
    # plt.savefig(f"0 below{kernel[0]}-dilate{index}.png")
    # plt.clf()
    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(image)
    return components, small_mask, stats


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow("labeled.png", labeled_img)
    # cv2.waitKey()


# imshow_components(labels_im)


def read_image(args):
    img = cv2.imread(args.cptv)
    print("img", img.shape)
    # img[img > 30] += 60
    components, small_mask, stats = detect_objects(img)
    small_mask[small_mask > 0] = 255
    print("small mask type", small_mask.dtype)
    cv2.imshow("mask.png", np.uint8(small_mask))
    cv2.waitKey()

    imshow_components(small_mask)
    i = 0
    for stat in stats:
        i += 10
        if stat[4] > 20 * 20:
            cv2.rectangle(
                img,
                (stat[0], stat[1]),
                (stat[0] + stat[2], stat[1] + stat[3]),
                (i, 255, 0),
                3,
            )
            print("mas  stats are", stat[4])
    cv2.imshow("detected.png", np.uint8(img))
    cv2.waitKey()
    print("DETECTED")
    # cv2.imshow("otsus", otsus)
    #
    # cv2.waitKey(0)


def theshold_saliency(image, otsus=False, threshold=100, kernel=(15, 15)):
    image = np.uint8(image)
    # image = cv2.fastNlMeansDenoising(np.uint8(image), None)

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("morph.png", np.uint8(image))

    # image = cv2.GaussianBlur(image, kernel, 0)
    flags = cv2.THRESH_BINARY
    if otsus:
        flags += cv2.THRESH_OTSU
    _, image = cv2.threshold(image, threshold, 255, flags)
    cv2.imshow("threshold.png", np.uint8(image))

    # image = cv2.dilate(image, kernel, iterations=1)

    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # import matplotlib.pyplot as plt
    #
    # imgplot = plt.imshow(image)
    # plt.savefig(f"0 below{kernel[0]}-dilate{index}.png")
    # plt.clf()
    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(image)
    return components, small_mask, stats


def read_avi(args):
    wait = 1
    vidcap = cv2.VideoCapture(args.cptv)
    count = 0
    fullbase = os.path.splitext(args.cptv)[0]
    start = False
    all_frames = []
    saliency = None
    while True:
        success, image = vidcap.read()
        if not success:
            break
            # if our saliency object is None, we need to instantiate it
        if saliency is None:
            saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
            saliency.setImagesize(image.shape[1], image.shape[0])
            saliency.init()

        count += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        repeats = 1
        if count < 8:
            repeats = 8
        print("count", count, len(all_frames))
        # elif count % 9 != 0:
        #     all_frames = []
        #     continue

        # detect_objects(otsus)

        for _ in range(repeats):
            (success, saliencyMap) = saliency.computeSaliency(gray)
            saliencyMap = (saliencyMap * 255).astype("uint8")
            # cv2.imshow("repeat.png", np.uint8(saliencyMap))
            # cv2.imshow("repeatImg.png", np.uint8(img))
            #
            # cv2.waitKey(100)
        # (success, saliencyMap) = saliency.computeSaliency(gray)
        if np.amin(saliencyMap) == 255:
            continue
        # saliencyMap = (saliencyMap * 255).astype("uint8")
        components, small_mask, stats = theshold_saliency(saliencyMap.copy())
        # print("running saliency", count, components)
        #
        rectangles = list(stats[1:])

        rect_i = 0
        while rect_i < len(rectangles):
            rect = rectangles[rect_i]
            merged = False
            mid_x = rect[2] / 2.0 + rect[0]
            mid_y = rect[3] / 2.0 + rect[1]
            index = 0
            while index < len(rectangles):
                r_2 = rectangles[index]
                if r_2[0] == rect[0]:
                    index += 1
                    continue
                r_mid_x = r_2[2] / 2.0 + r_2[0]
                r_mid_y = r_2[3] / 2.0 + r_2[1]
                distance = (mid_x - r_mid_x) ** 2 + (r_mid_y - mid_y) ** 2
                distance = distance**0.5
                distance = (
                    distance - max(rect[2], rect[3]) / 2.0 - max(r_2[2], r_2[3]) / 2.0
                )
                widest = max(rect[2], rect[3]) * 1
                within = r_2[0] > rect[0] and (r_2[0] + r_2[2]) <= (rect[0] + rect[2])
                within = (
                    within
                    and r_2[1] > rect[1]
                    and (r_2[1] + r_2[3]) <= (rect[1] + rect[3])
                )

                if distance < 40 or within:
                    print("merging", rect, r_2, distance, within)
                    rect[0] = min(rect[0], r_2[0])
                    rect[1] = min(rect[1], r_2[1])
                    rect[2] = max(rect[0] + rect[2], r_2[0] + r_2[2])
                    rect[3] = max(rect[1] + rect[3], r_2[1] + r_2[3])
                    rect[2] -= rect[0]
                    rect[3] -= rect[1]
                    print("second merged ", rect)
                    merged = True
                    # break
                    del rectangles[index]
                else:
                    index += 1
                    print("not mered", rect, r_2, distance)
            if merged:
                rect_i = 0
            else:
                rect_i += 1
        print("done merges")
        for stat in rectangles:
            print("stat is", stat)
            # if stat[4] < 5 * 5:
            # continue
            # print("final rect", stat)
            cv2.rectangle(
                gray,
                (stat[0], stat[1]),
                (stat[0] + stat[2], stat[1] + stat[3]),
                (0, 255, 0),
                3,
            )
        cv2.imshow("detected.png", np.uint8(gray))
        cv2.imshow("salicency.png", np.uint8(saliencyMap))

        cv2.imshow("contours.png", np.uint8(gray))

        imshow_components(small_mask)
        if wait == -1:
            key = cv2.waitKey()
        else:
            key = cv2.waitKey(wait)
        if key == 115:
            if wait == -1:
                wait = 1
            else:
                wait = -1
        # cv2.imshow("small mask.png", np.uint8(small_mask))

        continue

    return
    background = all_frames[-1]
    cv2.imwrite(f"{fullbase}-backgorund.png", background)  # save frame as JPEG file

    backsubtracted = np.float32(all_frames[8])
    # background
    backsubtracted[backsubtracted < 0] = 0

    a_max = np.amax(backsubtracted)
    a_min = np.amin(backsubtracted)

    backsubtracted = 255 * (backsubtracted - a_min) / (a_max - a_min)
    print(backsubtracted)

    mean = np.mean(backsubtracted)
    # backsubtracted[backsubtracted > 30] += 30
    # print(backsubtracted)
    #
    # cv2.imshow("frame.png", np.uint8(backsubtracted))
    # cv2.waitKey()
    # # return
    # cv2.imwrite(
    #     f"{fullbase}-backsubtracted.png", backsubtracted
    # )  # save frame as JPEG file
    # # return
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2
    for frame in all_frames:
        orig = frame.copy()
        # frame = frame - background
        frame[frame < 0] = 0
        a_max = np.amax(frame)
        a_min = np.amin(frame)
        # frame = 255 * (frame - a_min) / (a_max - a_min)
        frame[frame > 30] += 30

        i = 0
        print("detecting again")
        components, small_mask, stats = detect_objects(frame)
        # imshow_components(small_mask)

        for stat in stats:
            # i += 10
            if stat[4] > 20 * 20:
                cv2.rectangle(
                    orig,
                    (stat[0], stat[1]),
                    (stat[0] + stat[2], stat[1] + stat[3]),
                    (i, 255, 0),
                    3,
                )

                cv2.putText(
                    orig,
                    f"{stat}",
                    (stat[0] - 20, stat[1]),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType,
                )
        # cv2.imshow("detected.png", np.uint8(orig))
        # cv2.waitKey()
        # if cv2.waitKey(100) & 0xFF == ord("q"):  # if press SPACE bar
        #     break


def track_avi(args):
    # Define an initial bounding box
    bbox = (287, 23, 86, 320)
    bbox = (400, 149, 200, 200)
    video = cv2.VideoCapture(args.cptv)
    ok, frame = video.read()

    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)
    tracker = cv2.TrackerCSRT_create()
    # Initialize tracker with first frame and bo"unding box
    tracker_type = "CSRT"
    ok = tracker.init(frame, bbox)
    while True:
        # Read a new frame
        ok, frame = video.read()
        print("GO?")
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(
                frame,
                "Tracking failure detected",
                (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

            # Display tracker type on frame
            cv2.putText(
                frame,
                tracker_type + " Tracker",
                (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (50, 170, 50),
                2,
            )

            # Display FPS on frame
            cv2.putText(
                frame,
                "FPS : " + str(int(fps)),
                (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (50, 170, 50),
                2,
            )
            # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        if cv2.waitKey(100) & 0xFF == ord("q"):  # if press SPACE bar
            break

    video.release()
    cv2.destroyAllWindows()


def background_sub(args):

    cap = cv2.VideoCapture(args.cptv)
    fgbg = cv2.createBackgroundSubtractorMOG2(10)

    # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    i = 0
    saliency = None
    while 1:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not ret:
            break
        i += 1
        if saliency is None:
            saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
            saliency.setImagesize(gray.shape[1], gray.shape[0])
            saliency.init()

        repeats = 1
        if i < 8:
            repeats = 8
        for _ in range(repeats):
            (success, saliencyMap) = saliency.computeSaliency(gray)
            saliencyMap = (saliencyMap * 255).astype("uint8")

        fgmask = fgbg.apply(frame)
        # backg = fgbg.getBackgroundImage()
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # _, thresh, _ = theshold_saliency(fgmask, 10)
        cv2.imshow("frame", fgmask)
        # cv2.imshow("back", backg)
        cv2.imshow("saliency", saliencyMap)

        # imshow_components(thresh)
        # cv2.imshow("thresh", np.uint8(thresh))

        k = cv2.waitKey(100) & 0xFF

    cap.release()
    cv2.destroyAllWindows()


args = load_args()
ext = os.path.splitext(args.cptv)[1]
if ext != ".avi":
    # background_sub(args)
    read_image(args)
else:
    # background_sub(args)

    read_avi(args)
    # track_avi(args)
