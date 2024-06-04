"""
Synchronizing between stereo camera data and lidar data using GPS timestamp
"""
import numpy as np
from plyfile import PlyData
import os
import csv
from camera_lidar_projection import CameraLidarProjector
import cv2
import argparse

class CameraLidaSync(object):

    def __init__(self, camera_timestamps_csv: str = None, lidar_data_dir: str = None) -> None:
        """
        camera_timestamps_csv is the path to the csv file containing GPS time information for image.
            The file should have the content: imageId; timestamp. For example: 00000; 1684341237.5016508102
        lidar_data_dir is the path to folder containing lidar data in PLY format. 
            The folder should contain the information: lidar_id.ply. For example: 
            000000.ply; 000001.ply; 000002.ply; ...
        """
        
        self.__camera_timestamps_csv = np.genfromtxt(camera_timestamps_csv, delimiter=";")
        self.__camera_timestamps_csv = self.__camera_timestamps_csv[1:, :] # remove header data
        self.__camera_timestamps_csv[:, 0] = self.__camera_timestamps_csv[:, 0].astype(np.uint16)
        self.__camera_timestamps_csv_dict = {}

        for camera_data in self.__camera_timestamps_csv:
            img_id, img_t = camera_data
            self.__camera_timestamps_csv_dict[int(img_id)] = img_t

        self.__lidar_data_dir = lidar_data_dir
        self.__camera_timeshift = None
        self.__eps_timestamp_different = 0.002

    def __get_timestamp_ply(self, ply_file: str = None) -> list:
        """
        return the interval start and end time of the lidar scan
        """
        
        ply_data = PlyData.read(ply_file)
        data = ply_data.elements[0].data

        return [data[0][3], data[-1][3]]
    
    def __is_matched_lidar_camera(self, lidar_timestamp_interval: list, camera_timestamp: float,\
                                  matching_type="interval") -> bool:
        """
        matching_type = "interval" -> lidar_timestamp_interval[0] <= camera_timestamp and camera_timestamp <= lidar_timestamp_interval[1]
        """

        eps = self.__eps_timestamp_different

        if matching_type=="interval":
            return lidar_timestamp_interval[0] <= camera_timestamp and camera_timestamp <= lidar_timestamp_interval[1]
        elif matching_type=="first":
            if eps is None: print("Need to provide eps for matching_type=first")
            else: 
                lidar_tt_first = lidar_timestamp_interval[0]
                diff_ = abs(lidar_tt_first-camera_timestamp)
                return diff_ <= eps
        else:
            print("matching_type ", matching_type, " is not specified.")

    def __binary_search(self, start_idx_lidar: int, end_idx_lidar: int, timestamp: float) -> int:

        mid_idx = int((start_idx_lidar + end_idx_lidar) / 2)
        tmp = "%06d.ply" % mid_idx
        ply_file = os.path.join(self.__lidar_data_dir, tmp)

        lidar_timestamp_interval = self.__get_timestamp_ply(ply_file)

        if self.__is_matched_lidar_camera(lidar_timestamp_interval, timestamp, "first"):
            return mid_idx
        elif lidar_timestamp_interval[0] > timestamp:
            return self.__binary_search(start_idx_lidar, mid_idx, timestamp)
        else:
            return self.__binary_search(mid_idx, end_idx_lidar, timestamp)
    
    def __linear_search(self, start_idx_lidar: int, timestamp: float, search_threshold: int = 100) -> int:

        for i in range(search_threshold):
            
            tmp = "%06d.ply" % (start_idx_lidar + i)
            ply_file = os.path.join(self.__lidar_data_dir, tmp)
            lidar_timestamp_interval = self.__get_timestamp_ply(ply_file)

            if self.__is_matched_lidar_camera(lidar_timestamp_interval, timestamp, "first"):
                return start_idx_lidar + i
            
        return None # timestamp cannot be found
    
    def __seek_pivot_search_lidar_id_for_img_id(self, start_idx_lidar: int, timestamp: float) -> int:
        """
        Try to find first lidar_id having begining timestamp > img_id timestamp
        """
        
        while True:

            tmp = "%06d.ply" % (start_idx_lidar)
            ply_file = os.path.join(self.__lidar_data_dir, tmp)
            
            if not os.path.isfile(ply_file): return None

            lidar_timestamp_interval = self.__get_timestamp_ply(ply_file)
            lidar_timestamp_first = lidar_timestamp_interval[0]

            if lidar_timestamp_first < timestamp:
                start_idx_lidar += 1
            else:
                return start_idx_lidar

    def set_eps_timestamp_different(self, eps: float) -> None:
        self.__eps_timestamp_different = eps

    def set_camera_timeshift(self, camera_timeshift: float = None) -> None:

        self.__camera_timeshift = camera_timeshift

        self.__camera_timestamps_csv[:, 1] = self.__camera_timestamps_csv[:, 1] + camera_timeshift

        for i in self.__camera_timestamps_csv_dict.keys():
            self.__camera_timestamps_csv_dict[i] = self.__camera_timestamps_csv_dict[i] + camera_timeshift

    def show_lidar_timestamp_samples(self, num_samples: int = 10) -> None:
        
        for i in range(num_samples):

            ply_file = "%06d.ply" % i
            timestamp_ply = self.__get_timestamp_ply(os.path.join(self.__lidar_data_dir, ply_file))
            print("{} : {}".format(ply_file, timestamp_ply))

    def sync(self, start_idx_lidar: int, end_idx_lidar: int, save_file: str = None) -> None:
        
        matching_lidar_idx = None
        matching_result = []
        matching_result_headers = ["image_id", "lidar_scan_id"]

        for i, camera_data in enumerate(self.__camera_timestamps_csv):

            img_id = int(camera_data[0])
            img_timestamp = camera_data[1]

            if i == 0:
                matching_lidar_idx = self.__binary_search(start_idx_lidar, end_idx_lidar, img_timestamp)
            else:
                curr_matching_lidar_idx = self.__linear_search(matching_lidar_idx, img_timestamp)

                if curr_matching_lidar_idx is not None:
                    matching_lidar_idx = curr_matching_lidar_idx
                else:
                    print(img_id, " cannot be found")
            
            if save_file is not None:
                tmp = {}
                tmp["image_id"] = img_id
                tmp["lidar_scan_id"] = matching_lidar_idx
                matching_result.append(tmp)
        
        if save_file is not None:

            with open(save_file, "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=matching_result_headers)
                writer.writeheader()
                writer.writerows(matching_result)

    def sync_manual(self, img_idx: int, start_idx_lidar: int, end_idx_lidar: int, save_file: str = None) -> None:
        
        matching_lidar_idx = None
        matching_result = []
        matching_result_headers = ["image_id", "lidar_scan_id"]
        current_img_idx = img_idx
        num_matching_img = 0
        num_img = 0

        while current_img_idx in self.__camera_timestamps_csv_dict.keys():

            img_id_ = current_img_idx
            img_timestamp = self.__camera_timestamps_csv_dict[img_id_]
            is_found = False

            if matching_lidar_idx is None:
                matching_lidar_idx = self.__binary_search(start_idx_lidar, end_idx_lidar, img_timestamp) # Binary search for the first image id
                is_found = True
            else:
                curr_matching_lidar_idx = self.__linear_search(matching_lidar_idx, img_timestamp) # Linear search for the second image id and so on

                if curr_matching_lidar_idx is not None:
                    matching_lidar_idx = curr_matching_lidar_idx
                    is_found = True
                else:
                    # print(img_id_, " cannot be found")
                    matching_lidar_idx = self.__seek_pivot_search_lidar_id_for_img_id(matching_lidar_idx, img_timestamp)

            current_img_idx += 1           
            num_img += 1
            if is_found : num_matching_img += 1

            if save_file is not None:
                tmp = {}
                tmp["image_id"] = img_id_

                if is_found:
                    tmp["lidar_scan_id"] = matching_lidar_idx
                else:
                    tmp["lidar_scan_id"] = -1
                matching_result.append(tmp)
        
        if save_file is not None:

            with open(save_file, "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=matching_result_headers)
                writer.writeheader()
                writer.writerows(matching_result)
        
        print("# matching image / total image: {} / {}".format(num_matching_img, num_img))

class CameraLidarSyncManual(object):

    def __init__(self, camera_timestamps_csv: str, image_data_dir: str, lidar_data_dir: str, \
                 camera_lidar_projector: CameraLidarProjector) -> None:
        """
        This class uses for manually picking suitable lidar scan for the first image in the bag file,
            based on projecting lidar scan on the image plane.
        camera_timestamps_csv is the path to the csv file containing GPS time information for image.
            The file should have the content: imageId; timestamp. For example: 00000; 1684341237.5016508102
        lidar_data_dir is the path to folder containing lidar data in PLY format. 
            The folder should contain the information: lidar_id.ply. For example: 
            000000.ply; 000001.ply; 000002.ply; ...
        """
        
        self.__camera_timestamps_csv = np.genfromtxt(camera_timestamps_csv, delimiter=";")
        self.__camera_timestamps_csv = self.__camera_timestamps_csv[1:, :] # remove header data
        self.__camera_timestamps_csv[:, 0] = self.__camera_timestamps_csv[:, 0].astype(np.uint16)
        self.__image_dir = image_data_dir
        self.__lidar_dir = lidar_data_dir
        self.__camera_lidar_projector = camera_lidar_projector

    def __get_timestamp_ply(self, ply_file: str = None) -> list:
        """
        return the interval start and end time of the lidar scan
        """
        
        ply_data = PlyData.read(ply_file)
        data = ply_data.elements[0].data

        return [data[0][3], data[-1][3]]
    
    def __get_lidar_img_projection(self, img_path: str, lidar_path: str) -> None:

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
        projected_right_img = self.__camera_lidar_projector.show_points_on_img(right_img, right_img_points, right_camera_points_depth, \
                                                                               2, 20, "projected_right_img.png")

        return origin_img, projected_right_img
    
    def show_lidar_summary(self, num_sample: int = 100) -> None:

        for i in range(num_sample):

            tmp_lidar = "%06d" % i
            lidar_path = os.path.join(self.__lidar_dir, "{}.ply".format(tmp_lidar))

            t_begin, t_end = self.__get_timestamp_ply(lidar_path)
            print(tmp_lidar, " : ", t_begin, " - ", t_end - t_begin)

    def show_camera_summary(self, num_sample: int = 100) -> None:

        for i in range(num_sample):

            tmp_img = "%05d" %  i
            tmp_img_after = "%05d" %  (i+1)
            _, t_0 = self.__camera_timestamps_csv[self.__camera_timestamps_csv[:, 0] == i][0] # need to optimize this code
            _, t_1 = self.__camera_timestamps_csv[self.__camera_timestamps_csv[:, 0] == i+1][0] # need to optimize this code

            print(tmp_img, " : ", t_0, " - ", t_1 - t_0)

    def manual_pick(self, img_id: int = 0, lidar_id: int = 0) -> None:
        """
        Using 'a' and 'd' key to choosing preceding and succeeding lidar data to show on the image plane
        Using 'w' and 's' key to choosing preceding and succeeding image data to show with the current lidar
        """

        origin_img = None
        projected_right_img = None

        while True:

            tmp_img = "%05d" % img_id
            img_path = os.path.join(self.__image_dir, "{}.png".format(tmp_img))
            tmp_lidar = "%06d" % lidar_id
            lidar_path = os.path.join(self.__lidar_dir, "{}.ply".format(tmp_lidar))

            if os.path.isfile(lidar_path) and os.path.isfile(img_path):
                origin_img, projected_right_img = self.__get_lidar_img_projection(img_path, lidar_path)

            cv2.imshow("projected_right_img", projected_right_img)
            cv2.imshow("origin image", origin_img)
            
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break

            if key == ord('a') and lidar_id > 0:
                lidar_id -= 1

            if key == ord('d'):
                lidar_id += 1

            if key == ord('w') and img_id > 0:
                img_id -= 1

            if key == ord('s'):
                img_id += 1
            
            if key == ord('c'):
                print("Suitable lidar id - image id : ", lidar_id, " - ", img_id)
                print("Lidar timestamp: ", self.__get_timestamp_ply(lidar_path))
                print("Image timestamp: ", self.__camera_timestamps_csv[self.__camera_timestamps_csv[:, 0] == img_id])

def main(eps_timestamp_diff: float, save_file: str) -> None:

    camera_timestamps_csv = "/media/tuan/Daten/mapathon/mapathon_dataset/scenario_1_2/camera/processed_data/2023_08_16_09_01_55_1/image_timestamp_lisdt.csv"
    lidar_data_dir = "/media/tuan/Daten/mapathon/mapathon_dataset/scenario_1_2/Hesai64/LOCAL"
    camera_lidar_sync = CameraLidaSync(camera_timestamps_csv, lidar_data_dir)

    # camera_lidar_sync.set_camera_timeshift(7828077.74834919) # This value is compute by subtraction between first GPS timestamp in bagfile and start time in bag file
    camera_lidar_sync.set_camera_timeshift(7828078.943195752) # This value is chosen by using manual_pick from CameraLidarSyncManual
    camera_lidar_sync.set_eps_timestamp_different(eps_timestamp_diff)
    camera_lidar_sync.sync_manual(22, 0, 28620, save_file)

def main_manual() -> None:
    
    img_data_dir = "/media/tuan/Daten/mapathon/mapathon_dataset/scenario_1_2/camera/processed_data/2023_08_16_09_01_55_1/right_camera"
    lidar_data_dir = "/media/tuan/Daten/mapathon/mapathon_dataset/scenario_1_2/Hesai64/LOCAL"
    camera_timestamps_csv = "/media/tuan/Daten/mapathon/mapathon_dataset/scenario_1_2/camera/processed_data/2023_08_16_09_01_55_1/image_timestamp_lisdt.csv"
    camera_lidar_projector = CameraLidarProjector()
    camera_lidar_projector.load_calibration_data(camera_intrinsics_file="./calibration_data/intrinsics.txt", camera_extrinsics_file="./calibration_data/extrinsics.txt",\
                                                 cam_on_pcs_file="./calibration_data/mounting.txt", lidar_on_pcs_file="./calibration_data/Hesai64_on_PCS_mat.csv")
    
    camera_lidar_sync_manual = CameraLidarSyncManual(camera_timestamps_csv, img_data_dir, lidar_data_dir, camera_lidar_projector)
    camera_lidar_sync_manual.show_lidar_summary(num_sample=100); 
    camera_lidar_sync_manual.show_camera_summary(num_sample=100)
    camera_lidar_sync_manual.manual_pick(img_id=22, lidar_id=2090)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Argument for camera-lidar synchronization")

    parser.add_argument("--manual", action="store_true", help="Running manual camera-lidar synchronization")
    parser.add_argument("--eps_time_diff", type=float, default=0.003, help="Epsilon for defining timestamp different matching lidar and camera")
    parser.add_argument("--save_file", type=str, default="camera_lidar_sync.csv", help="File name for saving synchronized result")

    args = parser.parse_args()

    if args.manual:
        print("Running manual synchronization")
        main_manual()
    else:
        print("Running automatic synchronization with eps_time_diff = ", args.eps_time_diff, ". The result is saved to ", args.save_file)
        main(args.eps_time_diff, args.save_file)