"""
This code is used to compute relative orientation and translative between the stereo pair.
It includes modules:
* undistort image
* Compute Essential matrix using matching point
* Compute relative rotation matrix between the left image view and the right image view
* Compute the rotation matrix to the rectified plane (which is parallel to the baseline) R1, R2 for the left and the right view
* Draw the report to see the change of the angle computed by the above module
"""

from platform_calibration_loader import PlatformCalibrationLoader
import numpy as np
from typing import Tuple
import cv2
import argparse

class RelativeStereoComputation(object):

    def __init__(self, intrinsics_file: str, extrinsics_file: str) -> None:

        self.__platform_calibration_loader = PlatformCalibrationLoader()
        self.__platform_calibration_loader.parse_stereo_camera_calibration(intrinsics_file, extrinsics_file)
        self.__stereo_camera = self.__platform_calibration_loader.get_stereo_camera()

        self.__feature_descriptor = cv2.SIFT().create()
        self.__feature_matcher = cv2.BFMatcher()

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
                passed_ratio_test_pair.append([m])

        passed_ratio_kp_left = []
        passed_ratio_kp_right = []

        for p in passed_ratio_test_pair:
            passed_ratio_kp_left.append(kp_left[p.queryIdx])
            passed_ratio_kp_right.append(kp_right[p.trainIdx])
        
        passed_ratio_kp_left = np.float32(passed_ratio_kp_left).reshape(-1, 2)
        passed_ratio_kp_right = np.float32(passed_ratio_kp_right).reshape(-1, 2)

        return passed_ratio_kp_left, passed_ratio_kp_right

    def undistort_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        left_camera = self.__stereo_camera.get_left_camera()
        undistorted_left_image = cv2.undistort(left_camera, left_camera.get_camera_matrix(), left_camera.get_distortion_coeff())

        right_camera = self.__stereo_camera.get_right_camera()
        undistorted_right_image = cv2.undistort(right_image, right_camera.get_camera_matrix(), right_camera.get_distortion_coeff())

        return undistorted_left_image, undistorted_right_image

    def compute_essential_matrix_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:

        kp_left, kp_right = self.__find_matched_keypoints_stereo(left_image, right_image)

        left_camera = self.__stereo_camera.get_left_camera()
        right_camera = self.__stereo_camera.get_right_camera()

        essential_matrix, mask = cv2.findEssentialMat(kp_left, kp_right, cameraMatrix1=left_camera.get_camera_matrix(),\
                                                cameraMatrix2=right_camera.get_camera_matrix())
        
        print(mask.astype(np.float32).sum, " inliers found.")
        return essential_matrix
    
    # Return relative rotation matrix and relative translation vector
    def compute_relative_transformation_from_essential_matrix(self, essential_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    # Return rotation matrix R1, R2 to the rectified plane corresponding to the left and right image respectively
    def compute_rectified_rotation_matrix_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Argument for relative stereo computation.")

    parser.add_argument("--intrinsic_file", type=str, help="Path to the intrinsic file of stereo calibration.")
    parser.add_argument("--extrinsic_file", type=str, help="Path to the extrinsic file of stereo calibration.")
    parser.add_argument("--on_image", action="store_true", help="Running on the stereo image pair.")
    parser.add_argument("--left_image", type=str, help="Path to the left image.")
    parser.add_argument("--right_image", type=str, help="Path to the right image.")

    args = parser.parse_args()

    relative_stereo_computation = RelativeStereoComputation()