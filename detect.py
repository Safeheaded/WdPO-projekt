import json
from pathlib import Path
from typing import Dict

import click
import cv2
import numpy as np
from tqdm import tqdm

H_min = 20
S_min = 153
V_min = 100

H_max = 26
S_max = 255
V_max = 255

erosion = 5
dilation = 4

params = {
    "red": {
        "h_min": 174,
        "s_min": 174,
        "v_min": 108,
        "h_max": 179,
        "s_max": 226,
        "v_max": 215,
        "er": 3,
        "dil": 10
    },
    "yellow": {
        "h_min": 20,
        "s_min": 153,
        "v_min": 100,
        "h_max": 26,
        "s_max": 255,
        "v_max": 255,

        "er": 5,
        "dil": 4
    },
    "purple": {
        "h_min": 162,
        "s_min": 82,
        "v_min": 0,

        "h_max": 176,
        "s_max": 235,
        "v_max": 121,

        "er": 4,
        "dil": 12
    },
    "green": {
        "h_min": 36,
        "s_min": 195,
        "v_min": 137,

        "h_max": 51,
        "s_max": 255,
        "v_max": 245,

        "er": 1,
        "dil": 9
    }}


def empty_callback(value):
    pass


def manual_calibration(img_path: str, hsv_values):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    cv2.namedWindow('image')
    cv2.resizeWindow('image', 420, 375)

    h_min, s_min, v_min, h_max, s_max, v_max, er, dil = hsv_values.values()

    cv2.createTrackbar('H_min', 'image', h_min, 179, empty_callback)
    cv2.createTrackbar('S_min', 'image', s_min, 255, empty_callback)
    cv2.createTrackbar('V_min', 'image', v_min, 255, empty_callback)

    cv2.createTrackbar('H_max', 'image', h_max, 179, empty_callback)
    cv2.createTrackbar('S_max', 'image', s_max, 255, empty_callback)
    cv2.createTrackbar('V_max', 'image', v_max, 255, empty_callback)

    cv2.createTrackbar('Erosion', 'image', er, 20, empty_callback)
    cv2.createTrackbar('Dilation', 'image', dil, 20, empty_callback)

    # TODO: Implement detection method.
    scale = .2
    key = ord('a')
    while key != ord('q'):
        width = int(img.shape[0] * scale)
        height = int(img.shape[1] * scale)
        resized = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)

        H_min = cv2.getTrackbarPos('H_min', 'image')
        S_min = cv2.getTrackbarPos('S_min', 'image')
        V_min = cv2.getTrackbarPos('V_min', 'image')

        H_max = cv2.getTrackbarPos('H_max', 'image')
        S_max = cv2.getTrackbarPos('S_max', 'image')
        V_max = cv2.getTrackbarPos('V_max', 'image')

        erosion = cv2.getTrackbarPos('Erosion', 'image')
        dilation = cv2.getTrackbarPos('Dilation', 'image')

        res = count_snacks(img_path, {
            "h_min": H_min,
            "s_min": S_min,
            "v_min": V_min,
            "h_max": H_max,
            "s_max": S_max,
            "v_max": V_max,
            "er": erosion,
            "dil": dilation
        })
        cv2.drawContours(resized, res["contours"], -1, (0, 255, 0), 3)

        img_name = img_path.split('\\')[1]

        cv2.imshow('{} - result'.format(img_name), res["dilationRes"])
        cv2.imshow('{} - original'.format(img_name), resized)

        key = cv2.waitKey(30)

    cv2.destroyAllWindows()


def count_snacks(img_path: str, hsv_values):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h_min, s_min, v_min, h_max, s_max, v_max, er, dil = hsv_values.values()

    scale = .2
    width = int(img.shape[0] * scale)
    height = int(img.shape[1] * scale)
    resized = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)

    erosion_kernel = np.ones((er, er), np.uint8)
    dilation_kernel = np.ones((dil, dil), np.uint8)

    img_HSV = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    result = cv2.inRange(img_HSV, (h_min, s_min, v_min), (h_max, s_max, v_max))
    erosionRes = cv2.erode(result, erosion_kernel, iterations=1)
    dilationRes = cv2.dilate(erosionRes, dilation_kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilationRes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return {"contours": contours, "dilationRes": dilationRes}


def auto_check_color(img_path: str):
    results = {"red": 0, "green": 0, "yellow": 0, "purple": 0}
    global params
    for color in params:
        res = count_snacks(img_path, params[color])
        results[color] = len(res["contours"])

    return results


def check_validity(right_val, count, img_name, color):
    if right_val == count:
        print("{} - Success".format(color))
    else:
        print("{} - Should be {}, is {}".format(color, right_val, count))


def lets_rock(img_path: str, right_vals):
    results = {"red": 0, "green": 0, "yellow": 0, "purple": 0}
    colors = dict(red=(0, 0, 255),green=(0,255,0),yellow=(0,255,255),purple=(255,0,125))

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    scale = .2
    width = int(img.shape[0] * scale)
    height = int(img.shape[1] * scale)
    resized = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)
    cv2.namedWindow('result')
    cv2.imshow('result', resized)
    cv2.waitKey(200)

    global params
    for color in params:
        res = count_snacks(img_path, params[color])
        results[color] = len(res["contours"])
        cv2.drawContours(resized, res["contours"], -1, colors[color], 3)
        cv2.imshow('result', resized)
        cv2.waitKey(300)

    cv2.imshow('result', resized)
    cv2.waitKey(200)
    cv2.destroyAllWindows()

    return results


def detect(img_path: str) -> Dict[str, int]:

    results = {}
    global params
    # results = lets_rock(img_path, right_vals=right_vals)
    results = auto_check_color(img_path)
    # manual_calibration(img_path, params["purple"])

    return results


@click.command()
@click.option('-p', '--data_path', help='Path to data directory',
              type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    # with open("properResultValues.json", "r") as f:
    #     valid_data = json.load(f)
    #     # print(data[img_path.split("\\")[1]])

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    # score = 0
    #
    # for name in results:
    #     data = results[name]
    #     tmp_score = 0
    #     wage = 0
    #     print(name)
    #     for color in data:
    #         tmp_score += abs(valid_data[name][color] - data[color])
    #         wage += valid_data[name][color]
    #         check_validity(valid_data[name][color], data[color], name, color)
    #     print("\n")
    #
    #     score += tmp_score / wage
    #
    # score = score * 100/40
    # print(f"relative error: {score}%")
    # print(f"score: {100-score}%")

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
