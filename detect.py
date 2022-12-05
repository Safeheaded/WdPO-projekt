import json
from pathlib import Path
from typing import Dict

import click
import cv2
import numpy as np
from tqdm import tqdm


def empty_callback(value):
    pass


def detect(img_path: str) -> Dict[str, int]:
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
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    cv2.namedWindow('image')

    cv2.createTrackbar('H_min', 'image', 0, 179, empty_callback)
    cv2.createTrackbar('S_min', 'image', 0, 255, empty_callback)
    cv2.createTrackbar('V_min', 'image', 0, 255, empty_callback)

    cv2.createTrackbar('H_max', 'image', 0, 179, empty_callback)
    cv2.createTrackbar('S_max', 'image', 0, 255, empty_callback)
    cv2.createTrackbar('V_max', 'image', 0, 255, empty_callback)

    # TODO: Implement detection method.
    scale = .6
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

        img_HSV = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        test = cv2.inRange(img_HSV, (H_min, S_min, V_min), (H_max, S_max, V_max))
        # test = cv2.inRange(img_HSV, (59, 0, 0.992), (44, 0, .463))
        # back = cv2.cvtColor(test, cv2.COLOR_HSV2BGR)

        cv2.imshow('result', test)
        cv2.imshow('original', resized)

        key = cv2.waitKey(30)

    cv2.destroyAllWindows()

    red = 0
    yellow = 0
    green = 0
    purple = 0

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory',
              type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
