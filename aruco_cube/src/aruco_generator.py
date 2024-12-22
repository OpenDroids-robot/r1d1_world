#!/usr/bin/env python3

import click
import cv2
from cv2 import aruco
import numpy as np
import os
import rclpy
from rclpy.node import Node
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

class ArucoGenerator(Node):

    def __init__(self):
        super().__init__('aruco_creator')
        self.declare_parameter('dict', 'DICT_4X4_50')
        self.declare_parameter('id', 0)

        dict = self.get_parameter('dict').get_parameter_value().string_value
        self.marker_id = self.get_parameter('id').get_parameter_value().integer_value

        self.aruco_dict = aruco_dict_mapping.get(dict)

        self.get_logger().info(f'New aruco image with dictionary name {dict} and id {self.marker_id} saved')

    def create_marker(self, size, margin):

        # white background
        img = 255 * np.ones((size, size), dtype=np.uint8)
        img_marker = aruco.generateImageMarker(self.aruco_dict, self.marker_id, size - 2 * margin)

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
@click.argument("path", type=click.Path(exists=True))
@click.option("--tile_size", type=int, default=100)
def main(path, tile_size, args=None):
    rclpy.init(args=args)

    margin = int(0.3 * tile_size)

    marker_factory = ArucoGenerator()
    tile_map = TileMap(tile_size)

    for i in range(4):
        for j in range(3):
            if i != 1 and (j==0  or j == 2):
                continue
            if i==1 and j==1:
                marker_img = marker_factory.create_marker(tile_size, margin)
                tile_map.set_tile((i, j), marker_img)

    tile_img = tile_map.get_map_image()

    tile_img_square = np.zeros((tile_size * 4, tile_size*4))
    tile_img_square[:, (tile_size//2):(-tile_size//2)] = tile_img

    cv2.imwrite(os.path.join(path, "marker_tile.png"), tile_img)
    cv2.imwrite(os.path.join(path, "marker_tiles_square.png"), tile_img_square)

    marker_factory.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
