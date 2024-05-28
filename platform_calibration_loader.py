import numpy as np
import cv2

class MonoCamera(object):

    def __init__(self) -> None:
        
        self.__camera_matrix: np.ndarray = None
        self.__distortion_coeff: np.ndarray = None

    def set_camera_matrix(self, camera_matrix: np.ndarray) -> None:

        self.__camera_matrix = camera_matrix
    
    def set_distortion_coeff(self, distortion_coeff: np.ndarray) -> None:

        self.__distortion_coeff = distortion_coeff
    
    def get_camera_matrix(self) -> np.ndarray:
        return self.__camera_matrix
    
    def get_distortion_coeff(self) -> np.ndarray:
        return self.__distortion_coeff

class StereoCamera(object):

    def __init__(self) -> None:
        
        self.__left_camera = MonoCamera()
        self.__right_camera = MonoCamera()

        # relative orientation between left and right camera
        self.__relative_rotation: np.ndarray = None 
        self.__relative_translation: np.ndarray = None 
        self.__relative_orientation: np.ndarray = None
    
    def set_relative_rotation(self, relative_rotation: np.ndarray) -> None:

        self.__relative_rotation = relative_rotation
    
    def set_relative_translation(self, relative_translation: np.ndarray) -> None:

        self.__relative_translation = relative_translation

    def get_left_camera(self) -> None:
        return self.__left_camera

    def get_right_camera(self) -> None:
        return self.__right_camera
    
    def show_calibration_info(self) -> None:

        print("Left camera info: ")
        print(self.__left_camera.get_camera_matrix())
        print(self.__left_camera.get_distortion_coeff())

        print("Right camera info: ")
        print(self.__right_camera.get_camera_matrix())
        print(self.__right_camera.get_distortion_coeff())

        print("Relative orientation: ")
        print(self.__relative_rotation)
        print(self.__relative_translation)

class PlatformCalibrationLoader(object):
    """
    Class PlatformCalibrationLoader is used for managing the calibration data from 
    the multi-sensor platform, including: stereo camera, lidar and platform itself.
    """
    def __init__(self) -> None:
        
        self._stereo_camera = StereoCamera()

        # rotation matrix of the camera (left camera) in the platform coordinate system (pcs)
        self._camera_on_pcs_mat: np.ndarray = None
        # rotation matrix of the lidar in the platform coordinate system (pcs)
        self._lidar_on_pcs_mat: np.ndarray = None
    
    def set_camera_on_pcs_mat(self, camera_on_pcs_mat: np.ndarray) -> None:
        
        self._camera_on_pcs_mat = camera_on_pcs_mat

    def set_lidar_on_pcs_mat(self, lidar_on_pcs_mat: np.ndarray) -> None:

        self._lidar_on_pcs_mat = lidar_on_pcs_mat
    
    def parse_stereo_camera_calibration(self, intrinsic_camera_calibration_file: str, \
                                        extrinsic_camera_calibration_file: str) -> None:
        
        in_calib_data = cv2.FileStorage(intrinsic_camera_calibration_file, cv2.FILE_STORAGE_READ)
        ex_calib_data = cv2.FileStorage(extrinsic_camera_calibration_file, cv2.FILE_STORAGE_READ)

        self._stereo_camera.get_left_camera().set_camera_matrix(in_calib_data.getNode("M1").mat())
        self._stereo_camera.get_left_camera().set_distortion_coeff(in_calib_data.getNode("D1").mat())

        self._stereo_camera.get_right_camera().set_camera_matrix(in_calib_data.getNode("M2").mat())
        self._stereo_camera.get_right_camera().set_distortion_coeff(in_calib_data.getNode("D2").mat())

        self._stereo_camera.set_relative_rotation(ex_calib_data.getNode("R").mat())
        self._stereo_camera.set_relative_translation(ex_calib_data.getNode("T").mat())

    def parse_pcs_calibration(self, camera_on_pcs_calibration_file: str, \
                              lidar_on_pcs_calibration_file: str) -> None:
        pass

    def get_stereo_camera(self):

        return self._stereo_camera

if __name__=="__main__":

    platform_calibration_loader = PlatformCalibrationLoader()
    platform_calibration_loader.parse_stereo_camera_calibration("./calibration_data/intrinsics.txt", \
                                                                "./calibration_data/extrinsics.txt")
    platform_calibration_loader.get_stereo_camera().show_calibration_info()