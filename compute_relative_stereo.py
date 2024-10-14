"""
This code is used to compute relative orientation and translative between the stereo pair.
"""

from platform_calibration_loader import PlatformCalibrationLoader
import numpy as np
from typing import Tuple, List
import cv2
import argparse
import os
import matplotlib.pyplot as plt

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

    def __init__(self, intrinsics_file: str, extrinsics_file: str, viz_output_dir: str = None) -> None:

        self.__platform_calibration_loader = PlatformCalibrationLoader()
        self.__platform_calibration_loader.parse_stereo_camera_calibration(intrinsics_file, extrinsics_file)
        self.__stereo_camera = self.__platform_calibration_loader.get_stereo_camera()

        self.__feature_descriptor = cv2.SIFT().create()
        self.__feature_matcher = cv2.BFMatcher()

        # dependent view
        self.__euler_angle = IPI_OmFiKaRot()

        # two independent view
        self.p_relative_orientation_1 = IPI_OmFiKaRot()
        self.p_relative_orientation_2 = IPI_OmFiKaRot()

        if viz_output_dir is not None and not os.path.isdir(viz_output_dir):
            os.makedirs(viz_output_dir)

        self.__viz_output_dir = viz_output_dir
        self.__viz_img_idx = 0

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
        print("# Keypoints: ", passed_ratio_kp_left.shape, passed_ratio_kp_right.shape)

        return passed_ratio_kp_left, passed_ratio_kp_right

    def __convert_rad_to_gon(self, list_angle: list) -> float:

        list_result = []
        
        for angle in list_angle:
            list_result.append(angle*63.6619772368)
        return list_result
    
    def __draw_keypoints_on_image(self, image_left: np.ndarray, kp_coord_left: np.ndarray, mask_kp_left: np.ndarray,\
                                  image_right: np.ndarray, kp_coord_right: np.ndarray, mask_kp_right: np.ndarray, \
                                    visualize: bool) -> Tuple[np.ndarray, np.ndarray]:
        
        chosen_kp_coord_left  = kp_coord_left[mask_kp_left==1]
        chosen_kp_coord_right = kp_coord_right[mask_kp_right==1]
        # print(chosen_kp_coord_left, mask_kp_left, kp_coord_left.shape, mask_kp_left.shape); exit()

        rescale_value = 0.6

        rescale_image_left = cv2.resize(image_left, None, fx=rescale_value, fy=rescale_value)
        rescale_image_right = cv2.resize(image_right, None, fx=rescale_value, fy=rescale_value)
        kp_img = cv2.hconcat([rescale_image_left, rescale_image_right])
        matching_kp_img = cv2.hconcat([rescale_image_left, rescale_image_right])
        new_h, new_w, _ = rescale_image_left.shape

        for kp_left, kp_right in zip(chosen_kp_coord_left, chosen_kp_coord_right):

            b_val = 255 #np.random.randint(100, 255)
            g_val = 0 #np.random.randint(100, 255)
            r_val = 0 #np.random.randint(100, 255)

            _kp_left = np.int16(kp_left * rescale_value) 
            _kp_right = np.int16(kp_right * rescale_value)

            _kp_right[0] += new_w

            print("kp_left: ", _kp_left)
            print("kp_right: ", _kp_right)

            kp_img = cv2.circle(kp_img, _kp_left, 2, [b_val, g_val, r_val], -1)
            kp_img = cv2.circle(kp_img, _kp_right, 2, [b_val, g_val, r_val], -1)

            matching_kp_img = cv2.circle(matching_kp_img, _kp_left, 2, [b_val, g_val, r_val], -1)
            matching_kp_img = cv2.circle(matching_kp_img, _kp_right, 2, [b_val, g_val, r_val], -1)

            if _kp_left[0] <= new_w / 2:
                matching_kp_img = cv2.line(matching_kp_img, _kp_left, _kp_right, [0, 0, 255], 1)
            else:
                matching_kp_img = cv2.line(matching_kp_img, _kp_left, _kp_right, [0, 255, 0], 1)

        if visualize:
            cv2.imshow("Keypoint matching", matching_kp_img)
            cv2.waitKey(0)

        return kp_img, matching_kp_img

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

    # Compute relative rotation matrix for two images, then compute angles for each rotation matrix
    def compute_relative_transformation_stereo_5_angles(self, left_image: np.ndarray, right_image: np.ndarray) \
                            -> Tuple[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]:

        kp_left, kp_right = self.__find_matched_keypoints_stereo(left_image, right_image)
        print("kp_left shape: ", kp_left, kp_left.shape)

        left_camera = self.__stereo_camera.get_left_camera()
        right_camera = self.__stereo_camera.get_right_camera()

        retval, essential_matrix, relative_rotation, relative_translation, mask = cv2.recoverPose(points1=kp_left, points2=kp_right,
                                        cameraMatrix1=left_camera.get_camera_matrix(), distCoeffs1=left_camera.get_distortion_coeff(),
                                        cameraMatrix2=right_camera.get_camera_matrix(), distCoeffs2=right_camera.get_distortion_coeff()) 
        
        radiant_angle = self.__euler_angle.compute_parameters_from_rotation_matrix(relative_rotation, True)
        angles = self.__euler_angle.get_angles(in_degree=True)

        print("Angles: ", angles)

        # relative_translation[2] = -0.0003
        print(mask.astype(np.float32).sum(), " inliers found.")
        print("Relative translation: ", relative_translation.T, " . Baselength = ", np.sqrt(relative_translation[0] @ relative_translation[0].T))
        print("Relative_rotation: \n", relative_rotation, np.linalg.det(relative_rotation))

        R1, R2, P1, P2, Q, valid_pix_roi1, valid_pix_roi2 = cv2.stereoRectify(cameraMatrix1=left_camera.get_camera_matrix(), distCoeffs1=left_camera.get_distortion_coeff(),
                                                                              cameraMatrix2=right_camera.get_camera_matrix(), distCoeffs2=right_camera.get_distortion_coeff(),
                                                                              imageSize=left_camera.get_image_size_wh(), R=relative_rotation, \
                                                                                T=relative_translation)

        omega1, phi1, kappa1 = self.__convert_rad_to_gon(self.p_relative_orientation_1.compute_parameters_from_rotation_matrix(R1, True))
        omega2, phi2, kappa2 = self.__convert_rad_to_gon(self.p_relative_orientation_2.compute_parameters_from_rotation_matrix(R2, True))

        if abs(phi1) >= 3 or abs(phi2) >= 3:
            
            relative_5_angles = [omega1, phi1, kappa1, omega2, phi2, kappa2]
            _mask = np.squeeze(mask)
            kp_image, matching_kp_image = self.__draw_keypoints_on_image(left_image, kp_left, _mask, right_image, kp_right, _mask, visualize=False)

            if self.__viz_output_dir is not None:

                cv2.imwrite("{}/kp_image_{}.png".format(self.__viz_output_dir, self.__viz_img_idx), kp_image)
                cv2.imwrite("{}/matching_kp_image_{}.png".format(self.__viz_output_dir, self.__viz_img_idx), matching_kp_image)
                np.savetxt("{}/relative_orientation_5_angles_{}.txt".format(self.__viz_output_dir, self.__viz_img_idx), relative_5_angles)

                self.__viz_img_idx += 1

        return omega1, phi1, kappa1, omega2, phi2, kappa2, angles[0], angles[1], angles[2], relative_translation[0][0], \
            relative_translation[1][0], relative_translation[2][0]
    
    def test_compute_relative_transformation_stereo_5_angles(self):
        
        left_camera = self.__stereo_camera.get_left_camera()
        right_camera = self.__stereo_camera.get_right_camera()

        R1, R2, P1, P2, Q, valid_pix_roi1, valid_pix_roi2 = cv2.stereoRectify(cameraMatrix1=left_camera.get_camera_matrix(), distCoeffs1=left_camera.get_distortion_coeff(),\
                                                                              cameraMatrix2=right_camera.get_camera_matrix(), distCoeffs2=right_camera.get_distortion_coeff(),\
                                                                              imageSize=left_camera.get_image_size_wh(), R=self.__stereo_camera.get_relative_rotation(), \
                                                                              T=self.__stereo_camera.get_relative_translation())

        omega1, phi1, kappa1 = self.__convert_rad_to_gon(self.p_relative_orientation_1.compute_parameters_from_rotation_matrix(R1, True))
        omega2, phi2, kappa2 = self.__convert_rad_to_gon(self.p_relative_orientation_2.compute_parameters_from_rotation_matrix(R2, True))

        return omega1, phi1, kappa1, omega2, phi2, kappa2
    
    # Return omega, phi, kappa
    def compute_rotation_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.float32, np.float32, np.float32]:
        
        relative_rotation, relative_translation = self.compute_relative_transformation_stereo(left_image, right_image)

        radiant_angle = self.__euler_angle.compute_parameters_from_rotation_matrix(relative_rotation, True)
        return self.__euler_angle.get_angles(in_degree=True)

    def compute_rotation_stereo_sequence(self, left_image_dir: str, right_image_dir: str, start : int, step : int, num_samples: int) -> List:
        
        stats = []

        if num_samples > 0:

            for i in range(num_samples):

                image_id = i * step + start
                image_name = "image_{}".format(image_id)

                left_image_path = os.path.join(left_image_dir, "{}.png".format(image_name))
                right_image_path = os.path.join(right_image_dir, "{}.png".format(image_name))

                left_image = cv2.imread(left_image_path)
                right_image = cv2.imread(right_image_path)

                euler_angle = self.compute_rotation_stereo(left_image, right_image)
                print("{} : {}".format(image_id, euler_angle))
                stats.append(euler_angle)

        else:

            i = 0

            while True:

                image_id = i * step + start
                image_name = "image_{}".format(image_id)

                left_image_path = os.path.join(left_image_dir, "{}.png".format(image_name))
                right_image_path = os.path.join(right_image_dir, "{}.png".format(image_name))

                if not os.path.isfile(left_image_path) or not os.path.isfile(right_image_path):
                    print("{} or {} do not exists.".format(left_image_path, right_image_path))
                    break

                left_image = cv2.imread(left_image_path)
                right_image = cv2.imread(right_image_path)

                euler_angle = self.compute_rotation_stereo(left_image, right_image)
                print("{} : {}".format(image_id, euler_angle))
                stats.append(euler_angle)
        
        return stats

    def compute_rotation_stereo_sequence_5_angles(self, left_image_dir: str, right_image_dir: str, start : int, step : int, num_samples: int) -> List:
        
        stats = []

        if num_samples > 0:

            for i in range(num_samples):

                image_id = i * step + start
                image_name = "image_{}".format(image_id) #"%05d" % image_id
                # image_name = "%05d" % image_id

                left_image_path = os.path.join(left_image_dir, "{}.png".format(image_name))
                right_image_path = os.path.join(right_image_dir, "{}.png".format(image_name))

                left_image = cv2.imread(left_image_path)
                right_image = cv2.imread(right_image_path)

                euler_angle = self.compute_relative_transformation_stereo_5_angles(left_image, right_image)
                print("{} : {}".format(image_id, euler_angle))
                stats.append(euler_angle)

        else:

            i = 0

            while True:

                image_id = i * step + start
                image_name = "image_{}".format(image_id)

                left_image_path = os.path.join(left_image_dir, "{}.png".format(image_name))
                right_image_path = os.path.join(right_image_dir, "{}.png".format(image_name))

                if not os.path.isfile(left_image_path) or not os.path.isfile(right_image_path):
                    print("{} or {} do not exists.".format(left_image_path, right_image_path))
                    break

                left_image = cv2.imread(left_image_path)
                right_image = cv2.imread(right_image_path)

                euler_angle = self.compute_relative_transformation_stereo_5_angles(left_image, right_image)
                print("{} : {}".format(image_id, euler_angle))
                stats.append(euler_angle)

                i += 1
        
        return stats

    # To-do: Return rotation matrix R1, R2 to the rectified plane corresponding to the left and right image respectively
    def compute_rectified_rotation_matrix_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

class RelativeStereoStatistic(object):

    def __init__(self, statistic_file: str) -> None:

        self.__statistic_file = statistic_file
        self.__data = np.loadtxt(self.__statistic_file)

    def draw_euler_statistic(self, step: int = 2) -> None:

        idx = range(self.__data.shape[0]) 
        list_omega = self.__data[:, 0]
        list_phi = self.__data[:, 1]
        list_kappa = self.__data[:, 2]

        plt.plot(idx, list_omega, label = "Omega", linestyle="-.")
        plt.plot(idx, list_phi, label = "Phi", linestyle="--")
        plt.plot(idx, list_kappa, label = "Kappa", linestyle=":")
        plt.title("The variation of relative orientation between the left and right camera. Unit: degree")

        plt.legend()
        plt.show()

    def draw_euler_statistic_5_angles(self, step: int = 2) -> None:

        idx = range(self.__data.shape[0]) 
        list_omega_1 = self.__data[:, 0] - 0.081
        list_phi_1 = self.__data[:, 1] + 0.082
        list_kappa_1 = self.__data[:, 2] - 0.083

        list_omega_2 = self.__data[:, 3] + 0.081
        list_phi_2 = self.__data[:, 4] + 0.218
        list_kappa_2 = self.__data[:, 5] - 0.072

        threshold_value = 100

        chosen_idx1 = np.logical_and(np.logical_and((list_omega_1 < threshold_value),(list_phi_1 < threshold_value)), (list_kappa_1 < threshold_value))
        chosen_idx2 = np.logical_and(np.logical_and((list_omega_2 < threshold_value),(list_phi_2 < threshold_value)), (list_kappa_2 < threshold_value))
        chosen_idx = np.logical_and(chosen_idx1, chosen_idx2)
        idx = range(sum(chosen_idx))

        list_omega_1 = list_omega_1[chosen_idx]
        list_phi_1 = list_phi_1[chosen_idx]
        list_kappa_1 = list_kappa_1[chosen_idx]

        list_omega_2 = list_omega_2[chosen_idx]
        list_phi_2 = list_phi_2[chosen_idx]
        list_kappa_2 = list_kappa_2[chosen_idx]

        print("mean omega1, phi1, kappa1", np.mean(list_omega_1), np.mean(list_phi_1), np.mean(list_kappa_1))
        print("mean omega2, phi2, kappa2", np.mean(list_omega_2), np.mean(list_phi_2), np.mean(list_kappa_2))

        plt.plot(idx, list_omega_1, label = "Omega_1", linestyle="-.")
        plt.plot(idx, list_phi_1, label = "Phi_1", linestyle="--")
        plt.plot(idx, list_kappa_1, label = "Kappa_1", linestyle=":")

        plt.plot(idx, list_omega_2, label = "Omega_2", linestyle="-.")
        plt.plot(idx, list_phi_2, label = "Phi_2", linestyle="--")
        plt.plot(idx, list_kappa_2, label = "Kappa_2", linestyle=":")

        plt.title("The change of relative orientation between the left and right camera, compared to the calibrated parameters in the before calibration set. Unit: gon")

        plt.legend()
        plt.show()
        plt.close("all")


        list_relative_omega = self.__data[:, 6][chosen_idx] 
        list_relative_phi = self.__data[:, 7][chosen_idx]
        list_relative_kappa = self.__data[:, 8][chosen_idx]

        print("mean relative omega, phi, kappa", np.mean(list_relative_omega), np.mean(list_relative_phi), np.mean(list_relative_kappa))

        plt.plot(idx, list_relative_omega, label = "Omega", linestyle="-.")
        plt.plot(idx, list_relative_phi, label = "Phi", linestyle="--")
        plt.plot(idx, list_relative_kappa, label = "Kappa", linestyle=":")
        plt.title("Relative orientation between the left and right camera. Unit: degree")

        plt.legend()
        plt.show()
        plt.close("all")

        list_transition_x = self.__data[:, 9][chosen_idx] 
        list_transition_y = self.__data[:, 10][chosen_idx]
        list_transition_z = self.__data[:, 11][chosen_idx]

        print("mean relative transition x, y, z", np.mean(list_transition_x), np.mean(list_transition_y), np.mean(list_transition_z))

        plt.plot(idx, list_transition_x, label = "X", linestyle="-.")
        plt.plot(idx, list_transition_y, label = "Y", linestyle="--")
        plt.plot(idx, list_transition_z, label = "Z", linestyle=":")
        plt.title("Relative transition between the left and right camera. Unit: m")

        plt.legend()
        plt.show()
        plt.close("all")

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Argument for relative stereo computation.")

    parser.add_argument("--on_sequence", action="store_true", help="Running on the stereo image pair.")
    parser.add_argument("--intrinsic_file", type=str, help="Path to the intrinsic file of stereo calibration.")
    parser.add_argument("--extrinsic_file", type=str, help="Path to the extrinsic file of stereo calibration.")
    parser.add_argument("--left_image_dir", type=str, help="Path to the left image directory.")
    parser.add_argument("--right_image_dir", type=str, help="Path to the right image directory.")
    parser.add_argument("--start", type=int, default=0, help="Start id sample in stereo sequence, using to compute relative orientation.")
    parser.add_argument("--step", type=int, default=10, help="Step between each sample in stereo sequence, using to compute relative orientation.")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of stereo sample using for computing relative orientation")
    parser.add_argument("--saving_stats_file", type=str, default="stereo_relative_sequence.out", help="Name of file for saving relative orientation statistics.")

    parser.add_argument("--on_image", action="store_true", help="Running on the stereo image pair.")
    parser.add_argument("--left_image", type=str, help="Path to the left image.")
    parser.add_argument("--right_image", type=str, help="Path to the right image.")

    parser.add_argument("--draw_report", action="store_true", help="Drawing the report for relative orientation change.")

    args = parser.parse_args()

    if args.on_sequence:
        relative_stereo_computation = RelativeStereoComputation(args.intrinsic_file, args.extrinsic_file, "viz")
        stats = relative_stereo_computation.compute_rotation_stereo_sequence_5_angles(args.left_image_dir, args.right_image_dir, args.start, args.step, args.num_samples)
        np.savetxt(args.saving_stats_file, stats)
    
    if args.draw_report:
        drawer = RelativeStereoStatistic(args.saving_stats_file)
        drawer.draw_euler_statistic_5_angles()