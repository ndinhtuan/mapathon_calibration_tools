from camera_lidar_projection import CameraLidarProjector
import numpy as np
import os
import cv2
import argparse

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

    def show_camera_lidar_projection(self, video_name: str = None) -> None:

        if video_name is not None:
            out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), 3, (640, 480))
        
        for sync_data in self.__camera_lidar_sync_data:

            img_id, lidar_scan_id = int(sync_data[0]), int(sync_data[1])

            if lidar_scan_id == -1:
                continue

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
            origin_img = right_img.copy()
            projected_right_img = self.__camera_lidar_projector.show_points_on_img(right_img, right_img_points, right_camera_points_depth, 2, 20, "projected_right_img.png")

            if video_name is None:

                cv2.imshow("projected_right_img", projected_right_img)
                cv2.imshow("origin image", origin_img)

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            else:
                projected_right_img = cv2.resize(projected_right_img, (640, 480))
                out.write(projected_right_img)

        if video_name is not None :
            out.release()
        else:
            cv2.destroyAllWindows()

    def show_camera_lidar_projection_t_probe(self, video_name: str = None) -> None:

        if video_name is not None:
            out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), 3, (640, 480))
        
        for sync_data in self.__camera_lidar_sync_data:

            img_id, lidar_scan_id = int(sync_data[0]), int(sync_data[1])

            if lidar_scan_id == -1:
                continue

            tmp_img = "%05d" % img_id
            tmp_lidar = "%06d" % lidar_scan_id

            img_path = os.path.join(self.__image_dir, "{}.png".format(tmp_img))
            lidar_path = os.path.join(self.__lidar_dir, "{}.ply".format(tmp_lidar))
            self.__camera_lidar_projector.load_data(left_img_path=None, right_img_path=img_path, \
                                     left_lidar_path=None, right_lidar_path=lidar_path)
            
            right_camera_points = self.__camera_lidar_projector.transform_right_lidar_to_right_camera_coordinate_t_probe(is_debug=False)
            right_camera_points = right_camera_points[:-1, :].T
            right_camera_points = right_camera_points[right_camera_points[:, 2] > 0]
            right_camera_points_depth = np.linalg.norm(right_camera_points, axis=1).astype(np.uint8)

            right_img_points = self.__camera_lidar_projector.get_projected_right_points(right_camera_points)
            right_img_points = np.array(right_img_points)
            right_img_points = np.squeeze(right_img_points, 1)
            right_img_points = right_img_points.astype(np.int16)

            right_img = self.__camera_lidar_projector.get_right_img()
            origin_img = right_img.copy()
            projected_right_img = self.__camera_lidar_projector.show_points_on_img(right_img, right_img_points, right_camera_points_depth, 2, 20, "projected_right_img_tprobe.png")

            if video_name is None:

                cv2.imshow("projected_right_img_tprobe", projected_right_img)
                cv2.imshow("origin image", origin_img)

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            else:
                projected_right_img = cv2.resize(projected_right_img, (640, 480))
                out.write(projected_right_img)

        if video_name is not None :
            out.release()
        else:
            cv2.destroyAllWindows()

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Argument for camera-lidar synchronization")

    parser.add_argument("--t_probe", action="store_true", help="Running manual camera-lidar synchronization")

    args = parser.parse_args()

    img_data_dir = "/media/tuan/Daten/mapathon/mapathon_dataset/scenario_1_2/camera/processed_data/2023_08_16_09_01_55_1/right_camera"
    lidar_data_dir = "/media/tuan/Daten/mapathon/mapathon_dataset/scenario_1_2/Hesai64/LOCAL"
    camera_lidar_sync_file = "camera_lidar_sync.csv"
    camera_lidar_projector = CameraLidarProjector()
    camera_lidar_projector.load_calibration_data(camera_intrinsics_file="./calibration_data/intrinsics.txt", camera_extrinsics_file="./calibration_data/extrinsics.txt",\
                                                cam_on_pcs_file="./calibration_data/mounting.txt", lidar_on_pcs_file="./calibration_data/Hesai64_on_PCS_mat.csv",\
                                                cam_on_pcs_t_probe_file="./calibration_data/Cams_on_PCS_mat.csv")
    
    camera_lidar_verifier = CameraLidarVerify(img_data_dir, lidar_data_dir, camera_lidar_sync_file, camera_lidar_projector)

    if not args.t_probe:
        camera_lidar_verifier.show_camera_lidar_projection()
    else:
        camera_lidar_verifier.show_camera_lidar_projection_t_probe()