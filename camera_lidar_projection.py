import numpy as np
from plyfile import PlyData
import pandas as pd
import cv2

class CameraLidarProjector(object):

    def __init__(self) -> None:
        
        self._img_data : np.ndarray = None
        self._lidar_data : np.ndarray = None

        self._intrinsic_cam : np.ndarray = None
        self._distortion_coeff : np.ndarray = None
    
    @staticmethod
    def rotate_points(points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        pass
    
    def set_intrinsic_param(self, intrinsic_param: np.ndarray):
        self._intrinsic_cam = intrinsic_param
    
    def set_distortion_coeff(self, distortion_coeff: np.ndarray):
        self._distortion_coeff = distortion_coeff

    # add column 1 to the rightmost of the matrix
    def _transform_to_homogeneous_coordinate(self, matrix: np.ndarray) -> np.ndarray:
        
        r, c = matrix.shape
        tmp = np.ones(r)
        tmp = np.expand_dims(tmp, 1)
        matrix = np.hstack((matrix, tmp))

        return matrix

    def load_data(self, img_path: str, lidar_path: str) -> None:

        # reading lidar data
        ply_data = PlyData.read(lidar_path)
        data = ply_data.elements[0].data
        data_pd = pd.DataFrame(data)
        self._lidar_data = np.zeros(data_pd.shape, dtype=np.float64)
        property_names = data[0].dtype.names

        for i, name in enumerate(property_names):
            self._lidar_data[:, i] = data_pd[name]
        
        self._lidar_data = self._lidar_data[:, :3]
        self._lidar_data = self._transform_to_homogeneous_coordinate(self._lidar_data) 

    def get_projected_img(self) -> np.ndarray:
        
        assert self._intrinsic_cam is None, "self._intrinsic_cam is None"


if __name__=="__main__":

    camera_lidar_projector = CameraLidarProjector()
    camera_lidar_projector.load_data("./samples/image_511_r.png", "./samples/000000.ply")
    print(camera_lidar_projector._lidar_data.shape)

    right_intrinsic_camera = np.array([
        [-1.3666780120117235e+03, 0., 9.5231585161329167e+02], 
        [0., -1.3666559463125266e+03, 5.9523972910691634e+02], 
        [0., 0., 1.]
    ])

    right_distortion_coeff = np.array([
        -1.7464483294964098e-02, -1.6396647880186806e-02, -3.7916395337822523e-04, 
        3.7621485037772807e-04, -1.4324573971737545e-02])
    
    lidar_on_platform_rotation = np.array([
        [0.000697,-0.999962,0.008683,0.935490],
        [0.999920,0.000806,0.012599,-0.151705],
        [-0.012605,0.008673,0.999883,0.032571],
        [0.000000,0.000000,0.000000,1.000000]
    ])

    alpha, zeta, kappa = 3.1433163741851553e+00, 1.5698649441848733e+00, 1.5768085810372763e+00
    platform_calib = cv2.FileStorage("./calibration_data/mounting.txt", cv2.FILE_STORAGE_READ)
    print(platform_calib.getNode("angle3").real())