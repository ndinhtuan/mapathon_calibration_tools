"""
This code is used to compute relative orientation and translative between the stereo pair.
"""

from platform_calibration_loader import PlatformCalibrationLoader
import numpy as np
from typing import Tuple, List
import cv2
import argparse
import os

class IPI_OmFiKaRot(object):
    """
    Following to IPI definition and implementation

    Order of the angles
    p.x	= Omega  (o)
    p.y	= phi    (p)
    p.z = kappa  (k)
    
    Definition of the rotation matrix:
               (  cos(p)cos(k)                       -cos(p)sin(k)                        sin(p)       )
     R(o,p,k) = (  cos(o)sin(k)+sin(o)sin(p)cos(k)    cos(o)cos(k)-sin(o)sin(p)sin(k)     -sin(o)cos(p) )
                (  sin(o)sin(k)-cos(o)sin(p)cos(k)    sin(o)cos(k)+cos(o)sin(p)sin(k)      cos(o)cos(p) )
    """

    def __init__(self) -> None:
        
        self.__omega : np.float32 = None # in radiant
        self.__phi: np.float32 = None
        self.__kappa: np.float32 = None

        self.__rotation_matrix: np.ndarray = None

    def __convert_radiant_to_degree(self, angle: np.float32) -> np.float32:
        return angle * 180 / np.pi   

    def set_rotation_matrix(self, rotation_matrix: np.ndarray, use_ipi_coordinate_def: bool = True) -> None:

        self.__rotation_matrix = rotation_matrix

        if use_ipi_coordinate_def:
            m = np.array([
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0]
            ])

            self.__rotation_matrix = m @ (self.__rotation_matrix @ m)

    def get_angles(self, in_degree: bool = True) -> Tuple[np.float32, np.float32, np.float32]:
        
        if in_degree:
            return self.__convert_radiant_to_degree(self.__omega), self.__convert_radiant_to_degree(self.__phi), \
                self.__convert_radiant_to_degree(self.__kappa)
        else:
            return self.__omega, self.__phi, self.__kappa

    # Using implemtation from IPI library
    def compute_parameters_from_rotation_matrix(self, rotation_matrix: np.ndarray, use_ipi_coordinate_def: bool = True) \
                            -> Tuple[np.float32, np.float32, np.float32]:
        self.set_rotation_matrix(rotation_matrix, use_ipi_coordinate_def)

        self.__phi = np.arcsin(self.__rotation_matrix[0, 2])

        if np.cos(self.__phi) >= 0:

            self.__omega = np.arctan2(-self.__rotation_matrix[1, 2], self.__rotation_matrix[2, 2])
            self.__kappa = np.arctan2(-self.__rotation_matrix[0, 1], self.__rotation_matrix[0, 0])
        else:

            self.__omega = np.arctan2(self.__rotation_matrix[1, 2], -self.__rotation_matrix[2, 2])
            self.__kappa = np.arctan2(self.__rotation_matrix[0, 1], -self.__rotation_matrix[0, 0])

        return self.__omega, self.__phi, self.__kappa

class RelativeStereoComputation(object):

    def __init__(self, intrinsics_file: str, extrinsics_file: str) -> None:

        self.__platform_calibration_loader = PlatformCalibrationLoader()
        self.__platform_calibration_loader.parse_stereo_camera_calibration(intrinsics_file, extrinsics_file)
        self.__stereo_camera = self.__platform_calibration_loader.get_stereo_camera()

        self.__feature_descriptor = cv2.SIFT().create()
        self.__feature_matcher = cv2.BFMatcher()

        self.__euler_angle = IPI_OmFiKaRot()

    def __find_matched_keypoints_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function finds the matched keypoints in the left and right image.
        Return: keypoints in the left image, and the corresponding keypoints in the right image
        """
        
        gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        kp_left, des_left = self.__feature_descriptor.detectAndCompute(gray_left_image, None)
        kp_right, des_right = self.__feature_descriptor.detectAndCompute(gray_right_image, None)

        matches = self.__feature_matcher.knnMatch(des_left, des_right, k=2)

        passed_ratio_test_pair = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                passed_ratio_test_pair.append(m)

        passed_ratio_kp_left = []
        passed_ratio_kp_right = []

        for p in passed_ratio_test_pair:
            passed_ratio_kp_left.append(kp_left[p.queryIdx].pt)
            passed_ratio_kp_right.append(kp_right[p.trainIdx].pt)
        
        passed_ratio_kp_left = np.float32(passed_ratio_kp_left).reshape(-1, 2)
        passed_ratio_kp_right = np.float32(passed_ratio_kp_right).reshape(-1, 2)

        return passed_ratio_kp_left, passed_ratio_kp_right

    def undistort_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        left_camera = self.__stereo_camera.get_left_camera()
        undistorted_left_image = cv2.undistort(left_image, left_camera.get_camera_matrix(), left_camera.get_distortion_coeff())

        right_camera = self.__stereo_camera.get_right_camera()
        undistorted_right_image = cv2.undistort(right_image, right_camera.get_camera_matrix(), right_camera.get_distortion_coeff())

        return undistorted_left_image, undistorted_right_image

    def compute_essential_matrix_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        kp_left, kp_right = self.__find_matched_keypoints_stereo(left_image, right_image)

        left_camera = self.__stereo_camera.get_left_camera()
        right_camera = self.__stereo_camera.get_right_camera()

        essential_matrix, mask = cv2.findEssentialMat(kp_left, kp_right, cameraMatrix1=left_camera.get_camera_matrix(),\
                                                cameraMatrix2=right_camera.get_camera_matrix())
        
        print(mask.astype(np.float32).sum(), " inliers found.")
        return essential_matrix, mask
    
    # Return relative rotation matrix and relative translation vector
    def compute_relative_transformation_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        kp_left, kp_right = self.__find_matched_keypoints_stereo(left_image, right_image)

        left_camera = self.__stereo_camera.get_left_camera()
        right_camera = self.__stereo_camera.get_right_camera()

        retval, essential_matrix, relative_rotation, relative_translation, mask = cv2.recoverPose(points1=kp_left, points2=kp_right,
                                        cameraMatrix1=left_camera.get_camera_matrix(), distCoeffs1=left_camera.get_distortion_coeff(),
                                        cameraMatrix2=right_camera.get_camera_matrix(), distCoeffs2=right_camera.get_distortion_coeff()) 
        
        print(mask.astype(np.float32).sum(), " inliers found.")

        return relative_rotation, relative_translation
    
    # Return omega, phi, kappa
    def compute_rotation_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.float32, np.float32, np.float32]:
        
        relative_rotation, relative_translation = self.compute_relative_transformation_stereo(left_image, right_image)

        self.__euler_angle.compute_parameters_from_rotation_matrix(relative_rotation, True)
        return self.__euler_angle.get_angles(in_degree=True)

    def compute_rotation_stereo_sequence(self, left_image_dir: str, right_image_dir: str, step : int, num_samples: int) -> List:
        
        stats = []

        for i in range(num_samples):

            image_id = i * step
            image_name = "%05d" % image_id

            left_image_path = os.path.join(left_image_dir, "{}.png".format(image_name))
            right_image_path = os.path.join(right_image_dir, "{}.png".format(image_name))

            left_image = cv2.imread(left_image_path)
            right_image = cv2.imread(right_image_path)

            euler_angle = self.compute_rotation_stereo(left_image, right_image)
            print("{} : {}".format(image_id, euler_angle))
            stats.append(euler_angle)
        
        return stats

    # To-do: Return rotation matrix R1, R2 to the rectified plane corresponding to the left and right image respectively
    def compute_rectified_rotation_matrix_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Argument for relative stereo computation.")

    parser.add_argument("--intrinsic_file", type=str, help="Path to the intrinsic file of stereo calibration.")
    parser.add_argument("--extrinsic_file", type=str, help="Path to the extrinsic file of stereo calibration.")
    parser.add_argument("--left_image_dir", type=str, help="Path to the left image directory.")
    parser.add_argument("--right_image_dir", type=str, help="Path to the right image directory.")
    parser.add_argument("--step", type=int, default=10, help="Step between each sample in stereo sequence, using to compute relative orientation.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of stereo sample using for computing relative orientation")

    parser.add_argument("--on_image", action="store_true", help="Running on the stereo image pair.")
    parser.add_argument("--left_image", type=str, help="Path to the left image.")
    parser.add_argument("--right_image", type=str, help="Path to the right image.")

    args = parser.parse_args()

    relative_stereo_computation = RelativeStereoComputation(args.intrinsic_file, args.extrinsic_file)
    relative_stereo_computation.compute_rotation_stereo_sequence(args.left_image_dir, args.right_image_dir, args.step, args.num_samples)