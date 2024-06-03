from camera_lidar_projection import CameraLidarProjector
import numpy as np
import os
import cv2

class CameraLidarVerify(object):

    def __init__(self, image_dir: str, lidar_dir: str, camera_lidar_sync_file: str, \
                 camera_lidar_projector: CameraLidarProjector) -> None:
        
        self.__camera_lidar_projector = camera_lidar_projector
        self.__image_dir = image_dir
        self.__lidar_dir = lidar_dir 
        self.__camera_lidar_sync_file = camera_lidar_sync_file

        self.__camera_lidar_sync_data = np.genfromtxt(camera_lidar_sync_file, delimiter=",")
        self.__camera_lidar_sync_data = self.__camera_lidar_sync_data[1:, :]
        self.__camera_lidar_sync_data[:, 0] = self.__camera_lidar_sync_data[:, 0].astype(np.uint16)
        print(self.__camera_lidar_sync_data)

    def show_camera_lidar_projection(self):
        
        for sync_data in self.__camera_lidar_sync_data:

            img_id, lidar_scan_id = int(sync_data[0]), int(sync_data[1])
            tmp_img = "%05d" % img_id
            tmp_lidar = "%06d" % lidar_scan_id

            img_path = os.path.join(self.__image_dir, "{}.png".format(tmp_img))
            lidar_path = os.path.join(self.__lidar_dir, "{}.ply".format(tmp_lidar))
            self.__camera_lidar_projector.load_data(left_img_path=None, right_img_path=img_path, \
                                     left_lidar_path=None, right_lidar_path=lidar_path)
            
            right_camera_points = self.__camera_lidar_projector.transform_right_lidar_to_right_camera_coordinate(is_debug=False)
            right_camera_points = right_camera_points[:-1, :].T
            right_camera_points = right_camera_points[right_camera_points[:, 2] < 0]
            right_camera_points_depth = np.linalg.norm(right_camera_points, axis=1).astype(np.uint8)

            right_img_points = self.__camera_lidar_projector.get_projected_right_points(right_camera_points)
            right_img_points = np.array(right_img_points)
            right_img_points = np.squeeze(right_img_points, 1)
            right_img_points = right_img_points.astype(np.int16)

            right_img = self.__camera_lidar_projector.get_right_img()
            projected_right_img = self.__camera_lidar_projector.show_points_on_img(right_img, right_img_points, right_camera_points_depth, 2, 20, "projected_right_img.png")

            cv2.imshow("projected_right_img", projected_right_img)
            cv2.waitKey(0)

if __name__=="__main__":

    img_data_dir = "/media/tuan/Daten/mapathon/mapathon_dataset/scenario_1_2/camera/processed_data/2023_08_16_09_01_55_1/right_camera"
    lidar_data_dir = "/media/tuan/Daten/mapathon/mapathon_dataset/scenario_1_2/Hesai64/LOCAL"
    camera_lidar_sync_file = "camera_lidar_sync.csv"
    camera_lidar_projector = CameraLidarProjector()
    camera_lidar_projector.load_calibration_data(camera_intrinsics_file="./calibration_data/intrinsics.txt", camera_extrinsics_file="./calibration_data/extrinsics.txt",\
                                                 cam_on_pcs_file="./calibration_data/mounting.txt", lidar_on_pcs_file="./calibration_data/Hesai64_on_PCS_mat.csv")
    
    camera_lidar_verifier = CameraLidarVerify(img_data_dir, lidar_data_dir, camera_lidar_sync_file, camera_lidar_projector)
    camera_lidar_verifier.show_camera_lidar_projection()