from platform_calibration_loader import PlatformCalibrationLoader 
import cv2 
import numpy as np
import argparse
from typing import Tuple
import os
import glob

class StereoRectification(object):

    def __init__(self, intrinsics_file: str, extrinsics_file: str, img_size: tuple = None) -> None:
        
        self.__platform_calibration_loader = PlatformCalibrationLoader()
        self.__platform_calibration_loader.parse_stereo_camera_calibration(intrinsics_file, extrinsics_file)
        self.__stereo_camera = self.__platform_calibration_loader.get_stereo_camera()

        self.__img_size = img_size
    
    def load_stereo_camera_calibration(self, intrinsics_file: str, extrinsics_file: str) -> None:
        
        self.__platform_calibration_loader.parse_stereo_camera_calibration(intrinsics_file, extrinsics_file)
        self.__stereo_camera = self.__platform_calibration_loader.get_stereo_camera()

    # This function rectifies a image (left of right image in the stereo)
    def rectify_image(self, img: np.ndarray, is_left_img: bool) -> np.ndarray:
        
        cam_mat = None 
        dist_mat = None 
        R_mat = None 
        P_mat = None
        img_size = self.__img_size

        if is_left_img:
            cam_mat = self.__stereo_camera.get_left_camera().get_camera_matrix()
            dist_mat = self.__stereo_camera.get_left_camera().get_distortion_coeff()
            R_mat = self.__stereo_camera.get_left_camera().get_rectification_transform()
            P_mat = self.__stereo_camera.get_left_camera().get_projection_matrix()
        else:
            cam_mat = self.__stereo_camera.get_right_camera().get_camera_matrix()
            dist_mat = self.__stereo_camera.get_right_camera().get_distortion_coeff()
            R_mat = self.__stereo_camera.get_right_camera().get_rectification_transform()
            P_mat = self.__stereo_camera.get_right_camera().get_projection_matrix()
        
        map1, map2 = cv2.initUndistortRectifyMap(cam_mat, dist_mat, R_mat, P_mat, img_size, cv2.CV_16SC2)
        rectified_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        return rectified_img
    
    # This function rectifies a pair of correspondence image in the stereo camera setup
    def rectify_stereo_image(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        rectified_left_img = self.rectify_image(left_img, True)
        rectified_right_img = self.rectify_image(right_img, False)

        return rectified_left_img, rectified_right_img
    
    # This function rectifies stereo pair in `src_dir` directory, the rectified result are saved into
    # `dst_dir` directory. `src_dir` must include two sub-folder: "left_camera" and "right_camera" - 
    # containing the image taken from the left and the right camera
    def rectify_stereo_directory(self, src_dir, dst_dir) -> None:

        assert os.path.isdir(os.path.join(src_dir, "left_camera")) and os.path.isdir(os.path.join(src_dir, "right_camera")) \
            , "{} must contain two sub-folder: left_camera and right_camera".format(src_dir)

        assert not os.path.isdir(os.path.join(dst_dir, "left_camera")) and not os.path.isdir(os.path.join(dst_dir, "right_camera")) \
            , "{} folder is not empty. Please check and clean the destination folder.".format(dst_dir)

        os.makedirs(os.path.join(dst_dir, "left_camera"))
        os.makedirs(os.path.join(dst_dir, "right_camera"))

        self.set_image_size(os.path.join(src_dir, "left_camera"))

        list_left_img_path = glob.glob("{}/*".format(os.path.join(src_dir, "left_camera")))

        for left_img_path in list_left_img_path:
            name_img = left_img_path.split("/")[-1]
            right_img_path = os.path.join(src_dir, "right_camera", name_img)

            if not os.path.isfile(right_img_path):
                print("{} does not exists.".format(right_img_path))
                continue

            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)

            rectified_left_img, rectified_right_img = self.rectify_stereo_image(left_img, right_img)

            cv2.imwrite(os.path.join(dst_dir, "left_camera", name_img), rectified_left_img)
            cv2.imwrite(os.path.join(dst_dir, "right_camera", name_img), rectified_right_img)

    # Set value for self.__img_size by reading image content in one image folder
    def set_image_size(self, img_dir: str) -> None:

        list_img_path = glob.glob("{}/*".format(img_dir))

        assert len(list_img_path) > 0, "The image folder need to contain at least one image."

        img_sample = cv2.imread(list_img_path[0])
        tmp = img_sample.shape
        self.__img_size = (tmp[1], tmp[0])


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Argument for stereo image rectification.")

    parser.add_argument("--intrinsic_file", type=str, default="./calibration_data/intrinsics.txt", help="Path to the intrinsic parameter of stereo camera.")
    parser.add_argument("--extrinsic_file", type=str, default="./calibration_data/extrinsics.txt", help="Path to the extrinsic parameter of stereo camera.")
    parser.add_argument("--src_raw_folder", type=str, help="Path to the raw folder containing left and right raw image.")
    parser.add_argument("--dst_folder", type=str, help="Path to the folder will save the left and right rectified image.")

    args = parser.parse_args()

    stereo_rectification = StereoRectification(args.intrinsic_file, args.extrinsic_file)
    stereo_rectification.rectify_stereo_directory(args.src_raw_folder, args.dst_folder)