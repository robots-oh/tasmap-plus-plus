"""Points class i.e. point cloud."""
import numpy as np, struct


class Points:
    """Set of points defined by positions, colors, normals and more."""

    def __init__(self, positions, colors, normals, point_size, resolution, visible, alpha, shading_type=1):
        self.positions = positions
        self.colors = colors
        self.normals = normals
        self.point_size = point_size
        self.resolution = resolution
        self.visible = visible
        self.alpha = alpha
        self.shading_type = shading_type

    def get_properties(self, binary_filename):
        """
        :return: A dict conteining object properties. They are written into json and interpreted by javascript.
        """
        json_dict = {
            'type': 'points',
            'visible': self.visible,
            'alpha': self.alpha,
            'shading_type': self.shading_type,
            'point_size': self.point_size,
            'num_points': self.positions.shape[0],
            'binary_filename': binary_filename}
        return json_dict
    

    def write_binary(self, path, scale=0.001):

        pos16 = (self.positions / scale).astype(np.float16)      # 2B ×3
        nor16 = (self.normals * 32767).astype(np.int16)          # 2B ×3
        col8  = self.colors.astype(np.uint8)                     # 1B ×3

        with open(path, "wb") as f:
            f.write(pos16.tobytes())
            f.write(col8.tobytes())

    def write_blender(self, path):
        import open3d as o3d
        pcd_combined = o3d.geometry.TriangleMesh()

        def _create_sphere_at_xyz(xyz, colors, radius, resolution):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(colors)
            sphere = sphere.translate(xyz)
            return sphere

        # Make no-ob points whiter
        # pcd[np.all(pcd[:, 6:] == 120, axis=1), 6:] = [220., 220., 220.]

        for i in range(self.positions.shape[0]):
            pcd_combined += _create_sphere_at_xyz(self.positions[i],
                                                  self.colors[i] / 255.0,
                                                  self.point_size / 1000.0,
                                                  resolution=self.resolution)

        o3d.io.write_triangle_mesh(path, pcd_combined)
