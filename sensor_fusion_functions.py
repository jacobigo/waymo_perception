import gcsfs
from google.cloud import storage
import pyarrow.parquet as pq
import pyarrow.fs as pafs
import pandas as pd
import tensorflow as tf
import open3d as o3d
import numpy as np
import gc

#function imports
import semseg_functions


import os
os.environ["CLOUDSDK_CONFIG"] = "/home/jacob/.config/gcloud"

import subprocess
token = subprocess.check_output(
    ["/usr/bin/gcloud", "auth", "print-access-token"]
).decode().strip()



from datetime import datetime, timezone, timedelta
fs = pafs.GcsFileSystem(access_token=token, credential_token_expiration=datetime.now(timezone.utc) + timedelta(hours=1))


camera_img_files = fs.get_file_info(pafs.FileSelector("waymo_open_dataset_v_2_0_0/training/camera_image/"))
camera_calib_files = fs.get_file_info(pafs.FileSelector("waymo_open_dataset_v_2_0_0/training/camera_calibration/"))

camera_img_file = pq.ParquetFile(camera_img_files[0].path, filesystem=fs)
camera_img_rg_df = camera_img_file.read_row_group(0).to_pandas()
print(camera_img_files[0].path)
camera_img_rg_df.head(5)




#input in a certain timestamp's set of 5 camera values,
#along with the universal camera calibration data, 
#and the global LiDAR coordinates (for that timestamp)


#projection function

import io
from PIL import Image

def projection_onto_image(lidar_df, camera_df, lidar_calib_df, camera_calib_df, lidar_timestamp, camera_range=1):
    g_coords, masked_labels = semseg_functions.laser_process(laser_num=1, df=lidar_df, df_rgc=lidar_calib_df, df_seg=None, timestamp=lidar_timestamp)
    
    camera_tuples = []

    i = camera_range      #process 5 cameras per timestamp


    camera_extrensic_matrix = np.reshape(camera_calib_df["[CameraCalibrationComponent].extrinsic.transform"].loc[camera_calib_df["key.camera_name"] == i].iloc[0], (4, 4))

    #retreiving bytes of camera image for this timestamp
    camera_row_mask = (camera_df["key.camera_name"] == i) & (camera_df["key.frame_timestamp_micros"] == lidar_timestamp)
    camera_image = camera_df["[CameraImageComponent].image"].loc[camera_row_mask].iloc[0]
    image_bytes = io.BytesIO(camera_image)

    with Image.open(image_bytes) as img:
        camera_array = np.asarray(img.convert("RGB"))
    height, width = camera_array.shape[:2]

    g_coords_homo = np.column_stack((g_coords, np.ones(len(g_coords))))
    inv_camera_extrensic = np.linalg.inv(camera_extrensic_matrix)

    #putting 3D global points into 3D camera space
    g_coords_homo_T = np.transpose(g_coords_homo)
    camera_local_points_h_T = np.matmul(inv_camera_extrensic, g_coords_homo_T)
    camera_local_points_h = np.transpose(camera_local_points_h_T)

    camera_local_points = camera_local_points_h[:, :3]

    ## === REMEMBER ===
    ## In this dataset, the X axis is the plane pointing out of the camera; Z is vertical, Y is horizontal

    camera_mask_channel = camera_local_points[:, 0]
    camera_mask = camera_mask_channel > 0

    #apply the mask to only get X values greater than 0 (detected in front of the camera, not behind)
    camera_local_points_masked = camera_local_points[camera_mask]
    #camera_local_labels_masked = masked_labels[camera_mask]


    #retrive horizontal (u) and vertical (v) values for focal length and principal (center) point
    horiz_focal_length = camera_calib_df["[CameraCalibrationComponent].intrinsic.f_u"].loc[camera_calib_df["key.camera_name"] == i].iloc[0]
    vert_focal_length = camera_calib_df["[CameraCalibrationComponent].intrinsic.f_v"].loc[camera_calib_df["key.camera_name"] == i].iloc[0]
    horiz_center_point = camera_calib_df["[CameraCalibrationComponent].intrinsic.c_u"].loc[camera_calib_df["key.camera_name"] == i].iloc[0]
    vert_center_point = camera_calib_df["[CameraCalibrationComponent].intrinsic.c_v"].loc[camera_calib_df["key.camera_name"] == i].iloc[0]

    #slice to do perspective divide (divide by X, the depth in the image)
    Depth_X = camera_local_points_masked[:, 0]
    Left_Y = camera_local_points_masked[:, 1]
    Up_Z = camera_local_points_masked[:, 2]

    #pixel columns increase to the right and rows increase down, so we have to negate the left and up ratios
    Left_Y_div = (Left_Y / Depth_X) * -1
    Up_Z_div = (Up_Z / Depth_X) * -1

    #intrinsic transformations - scaling for focal and principal points (rounding (and casting to INT) for future indexing)
    Left_Y_col = np.round((Left_Y_div * horiz_focal_length) + horiz_center_point).astype(int)  #pixel column
    Up_Z_row = np.round((Up_Z_div * vert_focal_length) + vert_center_point).astype(int)      #pixel row

    
    #Out of Bounds masking: keeping points that only fall within the cameras FOV
    inbound_pixel_mask = (Left_Y_col >= 0) & (Left_Y_col < width) & (Up_Z_row >= 0) & (Up_Z_row < height)
    Left_Y_col_masked = Left_Y_col[inbound_pixel_mask]
    Up_Z_row_masked = Up_Z_row[inbound_pixel_mask]
    
    #apply the mask to the 3D camera points and labels, too
    #camera_pixel_labels = camera_local_labels_masked[inbound_pixel_mask]
    camera_pixel_3D_points = camera_local_points_masked[inbound_pixel_mask]

    rgb_values = camera_array[Up_Z_row_masked, Left_Y_col_masked]
    
    #returns for each camera, as a tuple in a list:
    #1. camera number
    #2. RGB color values for each point
    #3. pixel column array
    #4. pixel row array
    #5. 3D camera points
    #6. camera labels - EDIT: not using seg labels since they are too sparse when comparing with images - only using depth for classifying
    #7. raw JPEG bytes of the camera image (kept compressed to stay memory-light; decode on demand for overlays)
    camera_tuples.append((i, rgb_values, Left_Y_col_masked, Up_Z_row_masked, camera_pixel_3D_points, camera_image))
    
    return camera_tuples




#aligning the lidar, lidar_segmentation, and camera_image files and timestamps
#returns a list of tuples with 

def lidar_camera_processor(lidar_file, camera_file):


    # TIME_GAP_THRESH = 50000
    lidar_seg_camera = []
    
    lidar_calib_pq = pq.ParquetFile(f"waymo_open_dataset_v_2_0_0/training/lidar_calibration/{os.path.basename(lidar_file)}", filesystem=fs)
    lidar_calib_df = lidar_calib_pq.read_row_group(0).to_pandas()

    camera_calib_pq = pq.ParquetFile(f"waymo_open_dataset_v_2_0_0/training/camera_calibration/{os.path.basename(camera_file)}", filesystem=fs)
    camera_calib_df = camera_calib_pq.read_row_group(0).to_pandas()

    lidar_pq = pq.ParquetFile(lidar_file, filesystem=fs)
    camera_pq = pq.ParquetFile(camera_file, filesystem=fs)

    # for l_tup in lidar_timestamp_index:
    #     lidar_file, lidar_timestamp = l_tup
    #closest_stamp_gap = 100000000
        # for s_tup in seg_timestamp_index:

        #     seg_file, seg_timestamp = s_tup
        #     if seg_timestamp == lidar_timestamp and os.path.basename(lidar_file) == os.path.basename(seg_file):

        #lidar_df = semseg_functions.timestamp_aligner(lidar_file, lidar_timestamp)
        #     seg_df = semseg_functions.timestamp_aligner(seg_file, seg_timestamp)

    for j in range(lidar_pq.num_row_groups):
        lidar_df = lidar_pq.read_row_group(j).to_pandas()
        camera_df = camera_pq.read_row_group(j).to_pandas()
        camera_times = camera_pq.read_row_group(j, columns=["key.frame_timestamp_micros"]).to_pandas()
        
        # closest_stamp = 100000000
        # closest_file = None

        for lidar_timestamp in lidar_df["key.frame_timestamp_micros"].loc[lidar_df["key.laser_name"] == 1].unique():

            # camera_timestamp = camera_df["key.frame_timestamp_micros"].loc[camera_df["key.camera_name"] == 1].iloc[0]

            # if lidar_timestamp == camera_timestamp:
            if camera_times["key.frame_timestamp_micros"].isin([lidar_timestamp]).any():

                camera_tuple_list = projection_onto_image(lidar_df, camera_df, lidar_calib_df, camera_calib_df, lidar_timestamp, camera_range=1)
                lidar_seg_camera.append(camera_tuple_list)



    return lidar_seg_camera

            # current_stamp_gap = np.abs(lidar_timestamp - camera_timestamp)
            # if current_stamp_gap <= TIME_GAP_THRESH and current_stamp_gap <= closest_stamp_gap and os.path.basename(lidar_file) == os.path.basename(camera_file):
            #     #camera_calib_file = os.path.join("waymo_open_dataset_v_2_0_0/training/camera_calibration/", os.path.basename(camera_file))
            #     #camera_calib_df_t = timestamp_aligner(camera_calib_file, camera_timestamp)
            #     # camera_df = semseg_functions.timestamp_aligner(camera_file, camera_timestamp)
            #     closest_stamp_gap = current_stamp_gap
            #     closest_stamp = camera_timestamp
            #     closest_file = camera_file
            #     #continue

            # if lidar_timestamp - camera_timestamp <= 0 and closest_stamp_gap <= TIME_GAP_THRESH:
            #     break

    #     if closest_file is not None:
    #         #lidar_seg_camera.append((lidar_df, seg_df, camera_df))
    #         #begin processing projection
    #         camera_df = semseg_functions.timestamp_aligner(closest_file, closest_stamp)
    #         camera_tuple_list = projection_onto_image(lidar_df, camera_df, lidar_calib_df, camera_calib_df, lidar_timestamp, camera_range=[1])
    #         lidar_seg_camera.append(camera_tuple_list)

    
    # return lidar_seg_camera
    


lidar_timestamp_index = semseg_functions.folder_file_indexer("training/lidar", start_folder_index=0, end_folder_index=1)
#seg_timestamp_index = semseg_functions.folder_file_indexer("training/lidar_segmentation", start_folder_index=0, end_folder_index=1)
camera_timestamp_index = semseg_functions.folder_file_indexer("training/camera_image", start_folder_index=0, end_folder_index=1)

lidar_file = lidar_timestamp_index[0][0]
camera_file = camera_timestamp_index[0][0]

lidar_camera = lidar_camera_processor(lidar_file, camera_file)
print(len(lidar_camera))
