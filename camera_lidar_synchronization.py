"""
Synchronizing between stereo camera data and lidar data using GPS timestamp
"""
import numpy as np
from plyfile import PlyData
import os
import csv

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
        self.__lidar_data_dir = lidar_data_dir

        self.__camera_timeshift = None

    def __get_timestamp_ply(self, ply_file: str = None) -> list:
        """
        return the interval start and end time of the lidar scan
        """
        
        ply_data = PlyData.read(ply_file)
        data = ply_data.elements[0].data

        return [data[0][3], data[-1][3]]
    
    def __binary_search(self, start_idx_lidar: int, end_idx_lidar: int, timestamp: float) -> int:

        mid_idx = int((start_idx_lidar + end_idx_lidar) / 2)
        tmp = "%06d.ply" % mid_idx
        ply_file = os.path.join(self.__lidar_data_dir, tmp)

        lidar_timestamp_interval = self.__get_timestamp_ply(ply_file)

        if lidar_timestamp_interval[0] <= timestamp and timestamp <= lidar_timestamp_interval[1]:
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

            if lidar_timestamp_interval[0] <= timestamp and timestamp <= lidar_timestamp_interval[1]:
                return start_idx_lidar + i
            
        return None # timestamp cannot be found


    def set_camera_timeshift(self, camera_timeshift: float = None) -> None:

        self.__camera_timeshift = camera_timeshift

        self.__camera_timestamps_csv[:, 1] = self.__camera_timestamps_csv[:, 1] + camera_timeshift

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

if __name__=="__main__":

    camera_timestamps_csv = "/media/tuan/Daten/mapathon/mapathon_dataset/scenario_1_2/camera/processed_data/2023_08_16_09_01_55_1/image_timestamp_lisdt.csv"
    lidar_data_dir = "/media/tuan/Daten/mapathon/mapathon_dataset/scenario_1_2/Hesai64/LOCAL"
    camera_lidar_sync = CameraLidaSync(camera_timestamps_csv, lidar_data_dir)

    camera_lidar_sync.set_camera_timeshift(7828077.74834919)
    camera_lidar_sync.sync(0, 28620, "camera_liar_sync.csv")