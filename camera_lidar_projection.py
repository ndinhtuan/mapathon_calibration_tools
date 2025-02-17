import numpy as np
import open3d.visualization
from plyfile import PlyData
import pandas as pd
import cv2
from platform_calibration_loader import PlatformCalibrationLoader
import open3d

class CameraLidarProjector(object):

    def __init__(self) -> None:
        
        self.__left_img_data : np.ndarray = None
        self.__right_img_data : np.ndarray = None
        self.__left_lidar_data : np.ndarray = None
        self.__right_lidar_data : np.ndarray = None

        self.__platform_calibration_loader = PlatformCalibrationLoader()
    
    @staticmethod
    def rotate_points(points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        pass

    def __transform_points_from_lidar_to_pcs(self, lidar_points: np.ndarray) -> np.ndarray:

        transformation_matrix = self.__platform_calibration_loader.get_lidar_on_pcs_mat()
        transformed_points = transformation_matrix @ lidar_points
        return transformed_points
    
    def __transform_points_from_pcs_to_camera(self, pcs_points: np.ndarray) -> np.ndarray:

        transformation_matrix = self.__platform_calibration_loader.get_camera_on_pcs_mat() 
        return np.linalg.inv(transformation_matrix) @  pcs_points
    
    def __transform_points_from_pcs_to_right_camera_t_probe(self, pcs_points: np.ndarray) -> np.ndarray:

        right_camera_to_pcs_t_probe = self.__platform_calibration_loader.get_camera_on_pcs_mat_t_probe()
        return np.linalg.inv(right_camera_to_pcs_t_probe) @ pcs_points
    
    def __transform_points_from_left_to_right_camera(self, left_points: np.ndarray) -> np.ndarray:
        
        relative_orientation = self.__platform_calibration_loader.get_stereo_camera().get_relative_orientation()
        return np.linalg.inv(relative_orientation) @ left_points

    def __load_lidar_data(self, lidar_path: str) -> np.ndarray:
        
        # reading lidar data
        ply_data = PlyData.read(lidar_path)
        data = ply_data.elements[0].data
        data_pd = pd.DataFrame(data)
        lidar_data = np.zeros(data_pd.shape, dtype=np.float64)
        property_names = data[0].dtype.names

        for i, name in enumerate(property_names):
            lidar_data[:, i] = data_pd[name]
        
        lidar_data = lidar_data[:, :3]
        lidar_data = self.__transform_to_homogeneous_coordinate(lidar_data)

        self.__right_lidar_data= lidar_data.T 

        return self.__right_lidar_data

    def __load_img_data(self, img_path: str) -> np.ndarray:

        self.__right_img_data = cv2.imread(img_path)
        return self.__right_img_data

    # add column 1 to the rightmost of the matrix
    def __transform_to_homogeneous_coordinate(self, matrix: np.ndarray) -> np.ndarray:
        
        r, c = matrix.shape
        tmp = np.ones(r)
        tmp = np.expand_dims(tmp, 1)
        matrix = np.hstack((matrix, tmp))

        return matrix

    def get_right_img(self) -> np.ndarray:
        return self.__right_img_data
    
    def get_right_lidar_data(self) -> np.ndarray:
        return self.__right_lidar_data

    def load_calibration_data(self, camera_intrinsics_file: str, camera_extrinsics_file: str,\
                              cam_on_pcs_file: str, lidar_on_pcs_file: str, cam_on_pcs_t_probe_file: str=None) -> None:

        self.__platform_calibration_loader.parse_stereo_camera_calibration(camera_intrinsics_file, camera_extrinsics_file)
        self.__platform_calibration_loader.parse_pcs_calibration(cam_on_pcs_file, lidar_on_pcs_file)

        if cam_on_pcs_t_probe_file is not None:
            self.__platform_calibration_loader.parse_pcs_calibration_camera_t_probe(cam_on_pcs_t_probe_file)

    def load_data(self, left_img_path: str = None, right_img_path: str = None, \
                  left_lidar_path: str = None, right_lidar_path: str = None):

        if left_img_path is not None:
            self.__left_img_data = self.__load_img_data(left_img_path)

        if right_img_path is not None:
            self.__right_img_data = self.__load_img_data(right_img_path)

        if left_lidar_path is not None:
            self.__left_lidar_data = self.__load_lidar_data(left_lidar_path)
        
        if right_lidar_path is not None:
            self.__right_lidar_data = self.__load_lidar_data(right_lidar_path)

    def transform_right_lidar_to_right_camera_coordinate(self, is_debug: bool=True) -> np.ndarray:

        assert self.__right_lidar_data is not None, "self.__right_lidar_data is None" 
        assert self.__right_img_data is not None, "self.__right_img_data is None"

        if is_debug:
            print("Debugging point cloud in lidar coordinate system")
            _point_cloud = self.__right_lidar_data[:-1, :].T
            print(_point_cloud.shape)
            o3d_point_cloud = open3d.geometry.PointCloud()
            o3d_point_cloud.points = open3d.utility.Vector3dVector(_point_cloud)
            open3d.visualization.draw_geometries([o3d_point_cloud], window_name="lidar coordinate system")

        pcs_points = self.__transform_points_from_lidar_to_pcs(self.__right_lidar_data) 

        if is_debug:
            print("Debugging point cloud in platform coordinate system")
            _point_cloud = pcs_points[:-1, :].T
            print(_point_cloud.shape)
            o3d_point_cloud = open3d.geometry.PointCloud()
            o3d_point_cloud.points = open3d.utility.Vector3dVector(_point_cloud)
            open3d.visualization.draw_geometries([o3d_point_cloud], window_name="platform coordinate system")

        left_camera_points = self.__transform_points_from_pcs_to_camera(pcs_points)

        if is_debug:
            print("Debugging point cloud in left camera coordinate system")
            _point_cloud = left_camera_points[:-1, :].T
            print(_point_cloud.shape)
            o3d_point_cloud = open3d.geometry.PointCloud()
            o3d_point_cloud.points = open3d.utility.Vector3dVector(_point_cloud)
            open3d.visualization.draw_geometries([o3d_point_cloud], window_name="left camera coordinate system")

        right_camera_points = self.__transform_points_from_left_to_right_camera(left_camera_points)

        if is_debug:
            print("Debugging point cloud in left camera coordinate system")
            _point_cloud = right_camera_points[:-1, :].T
            print(_point_cloud.shape)
            o3d_point_cloud = open3d.geometry.PointCloud()
            o3d_point_cloud.points = open3d.utility.Vector3dVector(_point_cloud)
            open3d.visualization.draw_geometries([o3d_point_cloud], window_name="right camera coordinate system")

        return right_camera_points

    def transform_right_lidar_to_right_camera_coordinate_t_probe(self, is_debug: bool=True) -> np.ndarray:

        assert self.__right_lidar_data is not None, "self.__right_lidar_data is None" 
        assert self.__right_img_data is not None, "self.__right_img_data is None"

        if is_debug:
            print("Debugging point cloud in lidar coordinate system")
            _point_cloud = self.__right_lidar_data[:-1, :].T
            print(_point_cloud.shape)
            o3d_point_cloud = open3d.geometry.PointCloud()
            o3d_point_cloud.points = open3d.utility.Vector3dVector(_point_cloud)
            open3d.visualization.draw_geometries([o3d_point_cloud], window_name="lidar coordinate system")

        pcs_points = self.__transform_points_from_lidar_to_pcs(self.__right_lidar_data) 

        if is_debug:
            print("Debugging point cloud in platform coordinate system")
            _point_cloud = pcs_points[:-1, :].T
            print(_point_cloud.shape)
            o3d_point_cloud = open3d.geometry.PointCloud()
            o3d_point_cloud.points = open3d.utility.Vector3dVector(_point_cloud)
            open3d.visualization.draw_geometries([o3d_point_cloud], window_name="platform coordinate system")

        right_camera_points = self.__transform_points_from_pcs_to_right_camera_t_probe(pcs_points)

        if is_debug:
            print("Debugging point cloud in left camera coordinate system")
            _point_cloud = right_camera_points[:-1, :].T
            print(_point_cloud.shape)
            o3d_point_cloud = open3d.geometry.PointCloud()
            o3d_point_cloud.points = open3d.utility.Vector3dVector(_point_cloud)
            open3d.visualization.draw_geometries([o3d_point_cloud], window_name="right camera coordinate system")

        return right_camera_points

    def get_projected_right_points(self, points) -> np.ndarray:
        
        camera_matrix = self.__platform_calibration_loader.get_stereo_camera().get_right_camera().get_camera_matrix()
        distortion_coeff = self.__platform_calibration_loader.get_stereo_camera().get_right_camera().get_distortion_coeff()
        rvec = np.array([0., 0., 0.])
        tvec = np.array([0., 0., 0.])

        img_coord_point, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, distortion_coeff)
        return img_coord_point
    
    # list_points should be Nx2 dimention - N is a number of point
    def show_points_on_img(self, img: np.ndarray = None, list_points: np.ndarray = None, \
                           point_depth: np.ndarray = None, min_interested_depth: int = None, \
                            max_interested_depth: int = None, saved_path: str = None) -> None:

        point_depth = (((point_depth - min_interested_depth) / max_interested_depth) * 255).clip(0, 255).astype(np.uint8)
        depth2color = cv2.applyColorMap(point_depth, cv2.COLORMAP_TURBO)
        depth2color = np.squeeze(depth2color)

        for p, c in zip(list_points, depth2color):
            
            img = cv2.circle(img, p, 2, c.tolist(), -1)

        if saved_path is not None:
            cv2.imwrite(saved_path, img)
        
        return img

def main_ipi_calibration():

    camera_lidar_projector = CameraLidarProjector()
    camera_lidar_projector.load_data(left_img_path=None, right_img_path="./samples/image_511_r.png", \
                                     left_lidar_path=None, right_lidar_path="./samples/000000.ply")
    camera_lidar_projector.load_calibration_data(camera_intrinsics_file="./calibration_data/intrinsics.txt", camera_extrinsics_file="./calibration_data/extrinsics.txt",\
                                                 cam_on_pcs_file="./calibration_data/mounting.txt", lidar_on_pcs_file="./calibration_data/Hesai64_on_PCS_mat.csv")
    
    right_camera_points = camera_lidar_projector.transform_right_lidar_to_right_camera_coordinate(is_debug=False)
    right_camera_points = right_camera_points[:-1, :].T
    right_camera_points = right_camera_points[right_camera_points[:, 2] < 0]
    right_camera_points_depth = np.linalg.norm(right_camera_points, axis=1).astype(np.uint8)

    print(right_camera_points, right_camera_points.shape)

    right_img_points = camera_lidar_projector.get_projected_right_points(right_camera_points)
    right_img_points = np.array(right_img_points)
    right_img_points = np.squeeze(right_img_points, 1)
    right_img_points = right_img_points.astype(np.int16)

    print(right_img_points, right_img_points.shape, right_img_points.dtype)

    right_img = camera_lidar_projector.get_right_img()
    projected_right_img = camera_lidar_projector.show_points_on_img(right_img, right_img_points, right_camera_points_depth, 2, 20, "projected_right_img.png")

    cv2.imshow("projected_right_img", projected_right_img)
    cv2.waitKey(0)
    cv2.imwrite("projected_right_img.png", projected_right_img)

def main_t_probe():

    camera_lidar_projector = CameraLidarProjector()
    camera_lidar_projector.load_data(left_img_path=None, right_img_path="./samples/image_511_r.png", \
                                     left_lidar_path=None, right_lidar_path="./samples/000000.ply")
    camera_lidar_projector.load_calibration_data(camera_intrinsics_file="./calibration_data/intrinsics.txt", camera_extrinsics_file="./calibration_data/extrinsics.txt",\
                                                cam_on_pcs_file="./calibration_data/mounting.txt", lidar_on_pcs_file="./calibration_data/Hesai64_on_PCS_mat.csv",\
                                                cam_on_pcs_t_probe_file="./calibration_data/Cams_on_PCS_mat.csv")

    right_camera_points = camera_lidar_projector.transform_right_lidar_to_right_camera_coordinate_t_probe(is_debug=False)
    right_camera_points = right_camera_points[:-1, :].T
    right_camera_points = right_camera_points[right_camera_points[:, 2] > 0]
    right_camera_points_depth = np.linalg.norm(right_camera_points, axis=1).astype(np.uint8)

    print(right_camera_points, right_camera_points.shape)

    right_img_points = camera_lidar_projector.get_projected_right_points(right_camera_points)
    right_img_points = np.array(right_img_points)
    right_img_points = np.squeeze(right_img_points, 1)
    right_img_points = right_img_points.astype(np.int16)

    print(right_img_points, right_img_points.shape, right_img_points.dtype)

    right_img = camera_lidar_projector.get_right_img()
    projected_right_img = camera_lidar_projector.show_points_on_img(right_img, right_img_points, right_camera_points_depth, 2, 20, "projected_right_img.png")

    cv2.imshow("projected_right_img", projected_right_img)
    cv2.waitKey(0)

if __name__=="__main__":

    main_t_probe()