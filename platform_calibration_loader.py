import numpy as np
import cv2
import pandas as pd

class MonoCamera(object):

    def __init__(self) -> None:
        
        self.__camera_matrix: np.ndarray = None
        self.__distortion_coeff: np.ndarray = None

        # Attributes from stereo camera model of IPI library (based on opencv library)
        self.__rectification_transform: np.ndarray = None # following IPI definition (independent images 
                                        # with two angles (for left) and three angles (for right))
                                        # Notation: R1 (for left camera), R2 (for right camera)
        self.__projection_matrix: np.ndarray = None # projection matrix for rectified (epipolar) image
                                        # similar to definition of opencv as well as IPI

    def set_camera_matrix(self, camera_matrix: np.ndarray) -> None:
        self.__camera_matrix = camera_matrix
    
    def set_distortion_coeff(self, distortion_coeff: np.ndarray) -> None:
        self.__distortion_coeff = distortion_coeff

    def set_rectification_transform(self, rectification_transform: np.ndarray) -> None:
        self.__rectification_transform = rectification_transform

    def set_projection_matrix(self, projection_matrix: np.ndarray) -> None:
        self.__projection_matrix = projection_matrix
    
    def get_camera_matrix(self) -> np.ndarray:
        return self.__camera_matrix
    
    def get_distortion_coeff(self) -> np.ndarray:
        return self.__distortion_coeff
    
    def get_rectification_transform(self) -> np.ndarray:
        return self.__rectification_transform
    
    def get_projection_matrix(self) -> np.ndarray:
        return self.__projection_matrix

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

    def create_relative_orientation(self) -> None:

        self.__relative_orientation = np.hstack((self.__relative_rotation, self.__relative_translation))
        # homogeneous
        self.__relative_orientation = np.vstack((self.__relative_orientation, np.array([[0, 0, 0, 1]])))

    def get_relative_rotation(self) -> np.ndarray:
        return self.__relative_rotation
    
    def get_relative_translation(self) -> np.ndarray:
        return self.__relative_translation
    
    def get_relative_orientation(self) -> np.ndarray:
        return self.__relative_orientation

    def get_left_camera(self) -> MonoCamera:
        return self.__left_camera

    def get_right_camera(self) -> MonoCamera:
        return self.__right_camera
    
    def show_calibration_info(self) -> None:

        print("Left camera info: ")
        print(self.__left_camera.get_camera_matrix())
        print(self.__left_camera.get_distortion_coeff())

        print("Right camera info: ")
        print(self.__right_camera.get_camera_matrix())
        print(self.__right_camera.get_distortion_coeff())

        print("Relative orientation: ")
        print(self.__relative_orientation)

class PlatformCalibrationLoader(object):
    """
    Class PlatformCalibrationLoader is used for managing the calibration data from 
    the multi-sensor platform, including: stereo camera, lidar and platform itself.

    Current setup : 1 stereo camera and 1 lidar
    """
    def __init__(self) -> None:
        
        self.__stereo_camera = StereoCamera()

        # rotation matrix of the camera (left camera) in the platform coordinate system (pcs)
        self.__camera_on_pcs_mat: np.ndarray = None
        # rotation matrix of the lidar in the platform coordinate system (pcs)
        self.__lidar_on_pcs_mat: np.ndarray = None

        # rotation matrix of the camera (right camera) in the platform coordinate system (pcs)
        #   but using LaserScanner and T-proble in 3D lab to produce this matrix
        self.__camera_on_pcs_mat_t_probe: np.ndarray = None
    
    def set_camera_on_pcs_mat(self, camera_on_pcs_mat: np.ndarray) -> None:
        
        self.__camera_on_pcs_mat = camera_on_pcs_mat

    def set_lidar_on_pcs_mat(self, lidar_on_pcs_mat: np.ndarray) -> None:

        self.__lidar_on_pcs_mat = lidar_on_pcs_mat

    def set_camera_on_pcs_mat_t_probe(self, camera_on_pcs_mat_t_probe: np.ndarray) -> None:
        
        self.__camera_on_pcs_mat_t_probe = camera_on_pcs_mat_t_probe

    def get_stereo_camera(self) -> StereoCamera:
        return self.__stereo_camera

    def get_camera_on_pcs_mat(self) -> np.ndarray:
        return self.__camera_on_pcs_mat
    
    def get_lidar_on_pcs_mat(self) -> np.ndarray:
        return self.__lidar_on_pcs_mat
    
    def get_camera_on_pcs_mat_t_probe(self) -> np.ndarray:
        return self.__camera_on_pcs_mat_t_probe
    
    def parse_stereo_camera_calibration(self, intrinsic_camera_calibration_file: str, \
                                        extrinsic_camera_calibration_file: str) -> None:
        
        in_calib_data = cv2.FileStorage(intrinsic_camera_calibration_file, cv2.FILE_STORAGE_READ)
        ex_calib_data = cv2.FileStorage(extrinsic_camera_calibration_file, cv2.FILE_STORAGE_READ)

        self.__stereo_camera.get_left_camera().set_camera_matrix(in_calib_data.getNode("M1").mat())
        self.__stereo_camera.get_left_camera().set_distortion_coeff(in_calib_data.getNode("D1").mat())
        self.__stereo_camera.get_left_camera().set_rectification_transform(ex_calib_data.getNode("R1").mat())
        self.__stereo_camera.get_left_camera().set_projection_matrix(ex_calib_data.getNode("P1").mat())

        self.__stereo_camera.get_right_camera().set_camera_matrix(in_calib_data.getNode("M2").mat())
        self.__stereo_camera.get_right_camera().set_distortion_coeff(in_calib_data.getNode("D2").mat())
        self.__stereo_camera.get_right_camera().set_rectification_transform(ex_calib_data.getNode("R2").mat())
        self.__stereo_camera.get_right_camera().set_projection_matrix(ex_calib_data.getNode("P2").mat())

        self.__stereo_camera.set_relative_rotation(ex_calib_data.getNode("R").mat())
        self.__stereo_camera.set_relative_translation(ex_calib_data.getNode("T").mat())
        self.__stereo_camera.create_relative_orientation()
    
    @staticmethod
    # Function to create rotation matrix from AlZeKa angles and traslation vector
    def create_transformation_mat_from_IPIeuler(alpha: float, zeta: float, kappa: float, translation: np.ndarray) -> np.ndarray:
        
        a = alpha 
        z = zeta
        k = kappa
        cos = np.cos
        sin = np.sin
        
        R = np.zeros((3, 3))
        R[0, 0] = cos(a)*cos(z)*cos(k) - sin(a)*sin(k)
        R[0, 1] = -cos(a)*cos(z)*sin(k) - sin(a)*cos(k)
        R[0, 2] = cos(a)*sin(z)

        R[1, 0] = sin(a)*cos(z)*cos(k)+cos(a)*sin(k)
        R[1, 1] = -sin(a)*cos(z)*sin(k)+cos(a)*cos(k)
        R[1, 2] = sin(a)*sin(z)

        R[2, 0] = -sin(z)*cos(k)
        R[2, 1] = sin(z)*sin(k)
        R[2, 2] = cos(z)
        
        # M = np.array([
        #     [1.0, 0, 0],
        #     [0, 1.0, 0],
        #     [0, 0, -1.0]
        # ])
        # R = M @ R
        # R = R @ M

        translation = np.expand_dims(translation, 1)
        R = np.hstack((R, translation))
        tmp = np.array([0, 0, 0, 1])
        R = np.vstack((R, tmp))

        return R

    def parse_pcs_calibration(self, camera_on_pcs_calibration_file: str, \
                              lidar_on_pcs_calibration_file: str) -> None:

        # Parsing transformation matrix for camera on platform coordinate system
        cam_on_pcs_calib_data = cv2.FileStorage(camera_on_pcs_calibration_file, cv2.FILE_STORAGE_READ)

        alpha = cam_on_pcs_calib_data.getNode("angle1").real()
        zeta = cam_on_pcs_calib_data.getNode("angle2").real()
        kappa = cam_on_pcs_calib_data.getNode("angle3").real()
        translation_vect = np.array([cam_on_pcs_calib_data.getNode(i).real() for i in ["X0obj", "Y0obj", "Z0obj"]])

        cam_on_pcs_mat = PlatformCalibrationLoader.create_transformation_mat_from_IPIeuler(alpha, zeta, kappa, translation_vect)
        self.set_camera_on_pcs_mat(cam_on_pcs_mat)

        # Parsing transformation matrix for lidar on platform coordinate system
        lidar_on_pcs_mat_pd = pd.read_csv(lidar_on_pcs_calibration_file)
        lidar_on_pcs_mat = lidar_on_pcs_mat_pd.to_numpy()[:, :-1] # remove the last column - Timestamp
        self.set_lidar_on_pcs_mat(lidar_on_pcs_mat)

    # Parsing the transformation matrix from camera to platform, produced by LaserScan and T-probe
    def parse_pcs_calibration_camera_t_probe(self, camera_on_pcs_mat_t_probe_file: str) -> None:
        
        # Read content of the file, this file contains transformation matrix of both left and right camera
        #   on the platform coordinate system
        camera_on_pcs_mat_t_probe_pd = pd.read_csv(camera_on_pcs_mat_t_probe_file)
        right_camera_on_pcs_mat_t_probe = camera_on_pcs_mat_t_probe_pd.to_numpy()[4: , :][:, 1:-1]
        right_camera_on_pcs_mat_t_probe = right_camera_on_pcs_mat_t_probe.astype(np.float32)
        self.set_camera_on_pcs_mat_t_probe(right_camera_on_pcs_mat_t_probe)

    def show_sensor_transformation_info(self):

        print("Camera transformation matrix in PCS: ")
        print(self.__camera_on_pcs_mat)

        print("Lidar transformation matrix in PCS: ")
        print(self.__lidar_on_pcs_mat)

    def get_stereo_camera(self):

        return self.__stereo_camera

if __name__=="__main__":

    platform_calibration_loader = PlatformCalibrationLoader()
    platform_calibration_loader.parse_stereo_camera_calibration("./calibration_data/intrinsics.txt", \
                                                                "./calibration_data/extrinsics.txt")
    platform_calibration_loader.get_stereo_camera().show_calibration_info()

    platform_calibration_loader.parse_pcs_calibration("./calibration_data/mounting.txt",\
                                                      "./calibration_data/Hesai64_on_PCS_mat.csv")
    platform_calibration_loader.show_sensor_transformation_info()

    platform_calibration_loader.parse_pcs_calibration_camera_t_probe("./calibration_data/Cams_on_PCS_mat.csv")
    lidar_on_pcs_mat = platform_calibration_loader.get_camera_on_pcs_mat()
    camera_on_pcs_mat = platform_calibration_loader.get_camera_on_pcs_mat_t_probe()