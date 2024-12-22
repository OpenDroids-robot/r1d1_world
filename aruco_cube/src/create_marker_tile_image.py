import click
import cv2
from cv2 import aruco
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

marker_size = 300
aruco_dict_mapping = {
    'DICT_4X4_50': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    'DICT_4X4_100': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
    'DICT_4X4_250': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
    'DICT_4X4_1000': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),
    'DICT_5X5_50': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
    'DICT_5X5_100': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    'DICT_5X5_250': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
    'DICT_5X5_1000': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000),
    'DICT_6X6_50': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50),
    'DICT_6X6_100': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100),
    'DICT_6X6_250': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
    'DICT_6X6_1000': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000),
    'DICT_7X7_50': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50),
    'DICT_7X7_100': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100),
    'DICT_7X7_250': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250),
    'DICT_7X7_1000': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000),
    'DICT_ARUCO_ORIGINAL': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL),
    'DICT_APRILTAG_36h11': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
    'DICT_APRILTAG_25h9': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9),
}

class MarkerFactory:

    @staticmethod
    def create_marker(size, id, margin, dict):
        aruco_dict = aruco_dict_mapping.get(dict)

        # white background
        img = 255 * np.ones((size, size), dtype=np.uint8)
        img_marker = aruco.generateImageMarker(aruco_dict, id, size - 2 * margin)

        # add marker centered
        img[margin:-margin, margin:-margin] = img_marker

        return img


class TileMap:
    _map: np.ndarray

    def __init__(self, tile_size):
        self._map = 255 * np.ones((4, 3, tile_size, tile_size), dtype=np.uint8)

    def set_tile(self, pos: tuple, img: np.ndarray):
        assert np.all(self._map[pos[0], pos[1]].shape == img.shape)
        self._map[pos[0], pos[1]] = img

    def get_map_image(self):
        """ Merges the tile map into a single image """

        img = np.concatenate(self._map, axis=-1)
        img = np.concatenate(img, axis=-2)

        img = img.T

        return img


@click.command()
@click.option("--tile_size", type=int, default=100)
@click.option("--dict", type=str, default='DICT_4X4_50')
@click.option("--id", type=int, default=0)
def main(tile_size, id, dict):
    margin = int(0.3 * tile_size)

    image_path = '/home/ishan/OpenDroids_ws/src/aruco_cube/aruco_box/materials/textures'

    marker_factory = MarkerFactory()
    tile_map = TileMap(tile_size)

    marker_img = marker_factory.create_marker(tile_size, id, margin, dict)
    tile_map.set_tile((1, 1), marker_img)
                

    tile_img = tile_map.get_map_image()

    tile_img_square = np.zeros((tile_size * 4, tile_size*4))
    tile_img_square[:, (tile_size//2):(-tile_size//2)] = tile_img

    cv2.imwrite(os.path.join(image_path, "marker_tile.png"), tile_img)
    cv2.imwrite(os.path.join(image_path, "marker_tiles_square.png"), tile_img_square)
    print("Image generated")


if __name__ == '__main__':
    main()
