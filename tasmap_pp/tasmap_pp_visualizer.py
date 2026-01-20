"""The visualizer class is used to show 3d scenes."""
import sys
import os
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from utils.points import Points
from utils.cuboid import Cuboid
from utils.camera import Camera

import os
import sys
import shutil
import json
import numpy as np


class TASMapVisualizer:
    def __init__(self,
                 position: np.array = np.array([3.0, 3.0, 3.0]),
                 look_at: np.array = np.array([0.0, 0.0, 0.0]),
                 up: np.array = np.array([0.0, 0.0, 1.0]),
                 focal_length: float = 28.0):

        self.camera = Camera(
            position=np.array(position),
            look_at=np.array(look_at),
            up=np.array(up),
            focal_length=focal_length,
        )
        self.elements = {"Camera_0": self.camera}

    def __parse_name(self,
                     name: str) -> str:
        """Makes sure the name does not contain invalid character combinations.

        :param name:
        :return:
        """
        return name.replace(':', ';')

    def save(self,
            path: str,
            port: int=6010,
            verbose: bool=True) -> None:
        """Creates the visualization and displays the link to it.

        :param path: The path to save the visualization files.
        :param port: The port to show the visualization.
        :param verbose: Whether to print the web-server message or not.
        """

        # # Delete destination directory if it exists already
        directory_destination = os.path.abspath(path)
        if os.path.isdir(directory_destination):
            shutil.rmtree(directory_destination)

        # Copy website directory
        directory_source = os.path.realpath(os.path.join(os.path.dirname(__file__), "src"))
        shutil.copytree(directory_source, directory_destination, dirs_exist_ok=True)

        # Assemble binary data files
        nodes_dict = {}
        for name, e in self.elements.items():
            binary_file_path = os.path.join(directory_destination, name + ".bin")
            nodes_dict[name] = e.get_properties(name + ".bin")
            e.write_binary(binary_file_path)

        # Write json file containing all scene elements
        json_file = os.path.join(directory_destination, "nodes.json")
        with open(json_file, "w") as outfile:
            json.dump(nodes_dict, outfile, indent=4)

        # Display link
        if verbose:
          http_server_string = "python -m SimpleHTTPServer " + str(port)
          if sys.version[0] == "3":
              http_server_string = "python -m http.server " + str(port)
          print("")
          print(
              "************************************************************************"
          )
          print("1) Start local server:")
          print(f"   {http_server_string} --directory {directory_destination}")
          print(
              "************************************************************************"
          )



    def add_points(
        self,
        name: str,
        positions: np.array,
        colors: np.array=None,
        normals: np.array=None,
        point_size: int=25,
        resolution: int=3,
        visible: bool=True,
        alpha: float=1.0,
    ):
        """Add points to the visualizer.

        :param name: The name of the points displayed in the visualizer. Use ';' in the name to create sub-layers.
        :param positions: The point positions.
        :param normals: The point normals.
        :param colors: The point colors.
        :param point_size: The point size.
        :param resolution: The resolution of the blender sphere.
        :param visible: Bool if points are visible.
        :param alpha: Alpha value of colors.
        """

        assert positions.shape[1] == 3
        assert colors is None or positions.shape == colors.shape
        assert normals is None or positions.shape == normals.shape

        shading_type = 1  # Phong shading
        if colors is None:
            colors = np.ones(positions.shape, dtype=np.uint8) * 50  # gray
        if normals is None:
            normals = np.ones(positions.shape, dtype=np.float32)
            shading_type = 0  # Uniform shading when no normals are available

        positions = positions.astype(np.float32)
        colors = colors.astype(np.uint8)
        normals = normals.astype(np.float32)

        alpha = min(max(alpha, 0.0), 1.0)  # cap alpha to [0..1]

        self.elements[self.__parse_name(name)] = Points(
            positions, colors, normals, point_size, resolution, visible, alpha, shading_type
        )


    def add_cuboid(self,
                name: str,
                label: str,
                task:str,
                position: np.array,
                size: np.array,
                rotation: np.array,
                color: np.array = None,
                alpha: float = 1.0,
                edge_width: float = 1.0,
                visible: bool = True,
                label_visible: bool = True):
        """Add a cuboid (e.g., bounding box) to the visualizer.

        :param name: The name of the cuboid displayed in the visualizer. Use ';' in the name to create sub-layers.
        :param position: Center position of the cuboid (shape (3,)).
        :param size: Size of the cuboid (width, height, depth) (shape (3,)).
        :param rotation: Rotation of the cuboid (e.g., quaternion or Euler angles depending on system) (shape (4,) or (3,)).
        :param color: RGB color of the cuboid (shape (3,)), default gray if None.
        :param alpha: Transparency (0.0: transparent, 1.0: opaque).
        :param edge_width: Width of cuboid edges.
        :param visible: Whether the cuboid is visible.
        """

        assert position.shape == (3,), f"position should have shape (3,), got {position.shape}"
        assert size.shape == (3,), f"size should have shape (3,), got {size.shape}"
        assert rotation.shape in [(3,), (4,)], f"rotation should have shape (3,) or (4,), got {rotation.shape}"
        if color is None:
            color = np.array([128, 128, 128], dtype=np.uint8)  # default gray

        position = position.astype(np.float32)
        size = size.astype(np.float32)
        rotation = rotation.astype(np.float32)
        color = color.astype(np.uint8)

        alpha = min(max(alpha, 0.0), 1.0)  # cap alpha to [0..1]

        self.elements[self.__parse_name(name)] = Cuboid(
            label, task, position, size, rotation, color, alpha, edge_width, visible, label_visible
        )





