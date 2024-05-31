import numpy as np
from plyfile import PlyData
import pandas as pd
import cv2
from platform_calibration_loader import PlatformCalibrationLoader

class CameraLidarProjector(object):

    def __init__(self) -> None:
        
        self.__left_img_data : np.ndarray = None
        self.__right_img_data : np.ndarray = None
        self.__left_lidar_data : np.ndarray = None
        self.__right_lidar_data : np.ndarray = None

        self.__platform_calibration_loader = PlatformCalibrationLoader()
    
    @staticmethod
    def rotate_points(points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        pass

    def __transform_points_from_lidar_to_pcs(self, lidar_points: np.ndarray) -> np.ndarray:

        transformation_matrix = self.__platform_calibration_loader.get_lidar_on_pcs_mat()
        transformed_points = transformation_matrix @ lidar_points
        return transformed_points
    
    def __transform_points_from_pcs_to_camera(self, pcs_points: np.ndarray) -> np.ndarray:

        transformation_matrix = self.__platform_calibration_loader.get_camera_on_pcs_mat() 
        return np.linalg.inv(transformation_matrix) @  pcs_points
    
    def __transform_points_from_left_to_right_camera(self, left_points: np.ndarray) -> np.ndarray:
        
        relative_orientation = self.__platform_calibration_loader.get_stereo_camera().get_relative_orientation()
        return np.linalg.inv(relative_orientation) @ left_points

    def __load_lidar_data(self, lidar_path: str) -> np.ndarray:
        
        # reading lidar data
        ply_data = PlyData.read(lidar_path)
        data = ply_data.elements[0].data
        data_pd = pd.DataFrame(data)
        lidar_data = np.zeros(data_pd.shape, dtype=np.float64)
        property_names = data[0].dtype.names

        for i, name in enumerate(property_names):
            lidar_data[:, i] = data_pd[name]
        
        lidar_data = lidar_data[:, :3]
        lidar_data = self.__transform_to_homogeneous_coordinate(lidar_data) 

        return lidar_data.T

    def __load_img_data(self, img_path: str) -> np.ndarray:

        return cv2.imread(img_path)

    # add column 1 to the rightmost of the matrix
    def __transform_to_homogeneous_coordinate(self, matrix: np.ndarray) -> np.ndarray:
        
        r, c = matrix.shape
        tmp = np.ones(r)
        tmp = np.expand_dims(tmp, 1)
        matrix = np.hstack((matrix, tmp))

        return matrix

    def get_right_img(self) -> np.ndarray:
        return self.__right_img_data

    def load_calibration_data(self, camera_intrinsics_file: str, camera_extrinsics_file: str,\
                              cam_on_pcs_file: str, lidar_on_pcs_file: str) -> None:

        self.__platform_calibration_loader.parse_stereo_camera_calibration(camera_intrinsics_file, camera_extrinsics_file)
        self.__platform_calibration_loader.parse_pcs_calibration(cam_on_pcs_file, lidar_on_pcs_file)

    def load_data(self, left_img_path: str = None, right_img_path: str = None, \
                  left_lidar_path: str = None, right_lidar_path: str = None):

        if left_img_path is not None:
            self.__left_img_data = self.__load_img_data(left_img_path)

        if right_img_path is not None:
            self.__right_img_data = self.__load_img_data(right_img_path)

        if left_lidar_path is not None:
            self.__left_lidar_data = self.__load_lidar_data(left_lidar_path)
        
        if right_lidar_path is not None:
            self.__right_lidar_data = self.__load_lidar_data(right_lidar_path)

    def transform_right_lidar_to_right_camera_coordinate(self) -> np.ndarray:

        assert self.__right_lidar_data is not None, "self.__right_lidar_data is None" 
        assert self.__right_img_data is not None, "self.__right_img_data is None"

        pcs_points = self.__transform_points_from_lidar_to_pcs(self.__right_lidar_data) 
        left_camera_points = self.__transform_points_from_pcs_to_camera(pcs_points)
        right_camera_points = self.__transform_points_from_left_to_right_camera(left_camera_points)

        return right_camera_points

    def get_projected_right_points(self, points) -> np.ndarray:
        
        camera_matrix = self.__platform_calibration_loader.get_stereo_camera().get_right_camera().get_camera_matrix()
        distortion_coeff = self.__platform_calibration_loader.get_stereo_camera().get_right_camera().get_distortion_coeff()
        rvec = np.array([0., 0., 0.])
        tvec = np.array([0., 0., 0.])

        img_coord_point, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, distortion_coeff)
        return img_coord_point
    
    # list_points should be Nx2 dimention - N is a number of point
    def show_points_on_img(self, img: np.ndarray = None, list_points: np.ndarray = None, \
                           point_depth: np.ndarray = None, min_interested_depth: int = None, \
                            max_interested_depth: int = None, saved_path: str = None) -> None:

        point_depth = (((point_depth - min_interested_depth) / max_interested_depth) * 255).clip(0, 255).astype(np.uint8)
        print(point_depth, type(point_depth), point_depth.dtype)
        depth2color = cv2.applyColorMap(point_depth, cv2.COLORMAP_TURBO)
        depth2color = np.squeeze(depth2color)

        for p, c in zip(list_points, depth2color):

            img = cv2.circle(img, p, 2, c.tolist(), -1)

        if saved_path is not None:
            cv2.imwrite(saved_path, img)
        
        return img

if __name__=="__main__":

    camera_lidar_projector = CameraLidarProjector()
    camera_lidar_projector.load_data(left_img_path=None, right_img_path="./samples/image_511_r.png", \
                                     left_lidar_path=None, right_lidar_path="./samples/000000.ply")
    camera_lidar_projector.load_calibration_data(camera_intrinsics_file="./calibration_data/intrinsics.txt", camera_extrinsics_file="./calibration_data/extrinsics.txt",\
                                                 cam_on_pcs_file="./calibration_data/mounting.txt", lidar_on_pcs_file="./calibration_data/Hesai64_on_PCS_mat.csv")
    
    right_camera_points = camera_lidar_projector.transform_right_lidar_to_right_camera_coordinate()
    right_camera_points = right_camera_points[:-1, :].T
    right_camera_points = right_camera_points[right_camera_points[:, 0] > 0]
    right_camera_points_depth = np.linalg.norm(right_camera_points, axis=1).astype(np.uint8)

    print(right_camera_points, right_camera_points.shape)

    right_img_points = camera_lidar_projector.get_projected_right_points(right_camera_points)
    right_img_points = np.array(right_img_points)
    right_img_points = np.squeeze(right_img_points, 1)
    right_img_points = right_img_points.astype(int)

    print(right_img_points, right_img_points.shape)

    right_img = camera_lidar_projector.get_right_img()
    projected_right_img = camera_lidar_projector.show_points_on_img(right_img, right_img_points, right_camera_points_depth, 2, 20, "projected_right_img.png")

    cv2.imshow("projected_right_img", projected_right_img)
    cv2.waitKey(0)
