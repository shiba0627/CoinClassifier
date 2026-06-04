#import argparse
import math
import numpy as np
import cv2


def binalize(src_img):
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    gaus = cv2.GaussianBlur(gray, (15, 15), 5)
    bin = cv2.adaptiveThreshold(gaus, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, 2)
    bin2 = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return bin,bin2


def filter_object(bin_img, thresh_w, thresh_h, thresh_area):
    nlabels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(bin_img.astype(np.uint8))
    obj_stats_idx = np.where(
        (stats[1:, cv2.CC_STAT_WIDTH] > thresh_w[0])
        & (stats[1:, cv2.CC_STAT_WIDTH] < thresh_w[1])
        & (stats[1:, cv2.CC_STAT_HEIGHT] > thresh_h[0])
        & (stats[1:, cv2.CC_STAT_HEIGHT] < thresh_h[1])
        & (stats[1:, cv2.CC_STAT_AREA] > thresh_area[0])
        & (stats[1:, cv2.CC_STAT_AREA] < thresh_area[1])
    )
    return np.where(np.isin(labels_img - 1, obj_stats_idx), 255, 0).astype(np.uint8)


def filter_contours(bin_img, thresh_area):
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    new_cnt = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if thresh_area[0] > area or area > thresh_area[1]:
            continue
        (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = int(radius * radius * np.pi)
        if circle_area <= 0:
            continue
        area_diff = circle_area / area
        if 0.9 > area_diff or area_diff > 1.1:
            continue
        new_cnt.append(cnt)

    return new_cnt


def render_contours(contours, src_img):
    contours_img = np.copy(src_img)
    for cnt in contours:
        (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
        contours_img = cv2.circle(src_img, (int(center_x), int(center_y)), int(radius), (0, 0, 255), 8)
    
    return contours_img

#def parse_args() -> tuple:
#    parser = argparse.ArgumentParser()
#    parser.add_argument("IN_IMG", help="Input file")
#    parser.add_argument("OUT_IMG", help="Output file")
#    args = parser.parse_args()
#
#    return (args.IN_IMG, args.OUT_IMG)


def main() -> None:
    #(in_img, out_img) = parse_args()
    cap=cv2.VideoCapture(1)
    while True:
        ret,src_img=cap.read()
        if src_img is None:
            return
        height, width = src_img.shape[:2]
        bin,bin_img = binalize(src_img)

        bin=cv2.resize(bin,(int(width*0.2),int(height*0.2)))

        max_area = math.ceil((width * height) / 5)
        min_area = math.ceil((width * height) / 1000)
        bin_img = filter_object(bin_img, (0, (width / 2)), (0, (height / 2)), (min_area, max_area))

        #bin3_img=cv2.resize(bin_img,(int(width*0.2),int(height*0.2)))
        cv2.imshow('c',bin_img)

        contours = filter_contours(bin_img, (min_area, max_area))
        cnt_img = render_contours(contours, src_img)
        cv2.imshow('cnt',cnt_img)
        key=cv2.waitKey(10)
        if key==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()