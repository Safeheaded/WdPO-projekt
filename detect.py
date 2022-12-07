import json
from pathlib import Path
from typing import Dict

import click
import cv2
import numpy as np
from tqdm import tqdm


H_min = 0
S_min = 0
V_min = 0

H_max = 0
S_max = 0
V_max = 0

erosion = 0
dilation = 0

H_min = 174
S_min = 174
V_min = 108

H_max = 179
S_max = 226
V_max = 215

erosion = 3
dilation = 10

# żółty git

# H_min = 20
# S_min = 153
# V_min = 100
#
# H_max = 26
# S_max = 255
# V_max = 255
#
# erosion = 5
# dilation = 4

def empty_callback(value):
    pass


def manual_calibration(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    cv2.namedWindow('image')
    cv2.resizeWindow('image', 420, 375)

    global H_min, S_min, V_min, H_max, S_max, V_max, erosion, dilation

    cv2.createTrackbar('H_min', 'image', H_min, 179, empty_callback)
    cv2.createTrackbar('S_min', 'image', S_min, 255, empty_callback)
    cv2.createTrackbar('V_min', 'image', V_min, 255, empty_callback)

    cv2.createTrackbar('H_max', 'image', H_max, 179, empty_callback)
    cv2.createTrackbar('S_max', 'image', S_max, 255, empty_callback)
    cv2.createTrackbar('V_max', 'image', V_max, 255, empty_callback)

    cv2.createTrackbar('Erosion', 'image', erosion, 20, empty_callback)
    cv2.createTrackbar('Dilation', 'image', dilation, 20, empty_callback)

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

        erosion_kernel = np.ones((erosion, erosion), np.uint8)
        dilation_kernel = np.ones((dilation, dilation), np.uint8)

        img_HSV = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        result = cv2.inRange(img_HSV, (H_min, S_min, V_min), (H_max, S_max, V_max))
        erosionRes = cv2.erode(result, erosion_kernel, iterations=1)
        dilationRes = cv2.dilate(erosionRes, dilation_kernel, iterations=1)

        # ret, thresh = cv2.threshold(dilationRes, 127, 255, 0)
        contours, hierarchy = cv2.findContours(dilationRes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print(len(contours))
        cv2.drawContours(resized, contours, -1, (0, 255, 0), 3)

        img_name = img_path.split('\\')[1]

        cv2.imshow('{} - result'.format(img_name), dilationRes)
        cv2.imshow('{} - original'.format(img_name), resized)

        key = cv2.waitKey(30)

    cv2.destroyAllWindows()


def auto_check_color(img_path: str, color: str, right_vals):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    scale = .2
    width = int(img.shape[0] * scale)
    height = int(img.shape[1] * scale)
    resized = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)

    global H_min, S_min, V_min, H_max, S_max, V_max, erosion, dilation

    erosion_kernel = np.ones((erosion, erosion), np.uint8)
    dilation_kernel = np.ones((dilation, dilation), np.uint8)

    img_HSV = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    result = cv2.inRange(img_HSV, (H_min, S_min, V_min), (H_max, S_max, V_max))
    erosionRes = cv2.erode(result, erosion_kernel, iterations=1)
    dilationRes = cv2.dilate(erosionRes, dilation_kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilationRes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # img_name = img_path.split('\\')[1]
    #
    # if color == 'red':
    #     check_validity(right_vals[img_name]['red'], len(contours), img_name)
    # elif color == "purple":
    #     check_validity(right_vals[img_name]['purple'], len(contours), img_name)
    # elif color == "yellow":
    #     check_validity(right_vals[img_name]['yellow'], len(contours), img_name)
    # elif color == "green":
    #     check_validity(right_vals[img_name]['green'], len(contours), img_name)

    results = {'red': 0, 'yellow': 0, 'green': 0, 'purple': 0}

    if color == 'red':
        results["red"] = len(contours)
    elif color == "purple":
        results["purple"] = len(contours)
    elif color == "yellow":
        results["yellow"] = len(contours)
    elif color == "green":
        results["green"] = len(contours)

    return results


def check_validity(right_val, count, img_name):
    if right_val == count:
        print("{} - Success".format(img_name))
    else:
        print("{} - Should be {}, is {}".format(img_name, right_val, count))



def detect(img_path: str, rightVals: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """

    # results = auto_check_color(img_path, right_vals=rightVals, color="red")
    manual_calibration(img_path)
    red = 0
    yellow = 0
    green = 0
    purple = 0

    # return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}
    return {}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory',
              type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    with open("properResultValues.json", "r") as f:
        valid_data = json.load(f)
        # print(data[img_path.split("\\")[1]])

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path), valid_data)
        results[img_path.name] = fruits

    for name in results:
        data = results[name]
        if data["red"] == valid_data[name]["red"]:
            print("{} - Success".format(name))
        else:
            print("{} - Should be {}, is {}".format(name, valid_data[name]["red"], data["red"]))

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
