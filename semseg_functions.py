# Functions from the semseg.ipynb notebook for better reference between files
# By Jacob Igo
# Summer 2026


#library imports
import gcsfs
from google.cloud import storage
import pyarrow.parquet as pq
import pyarrow.fs as pafs
import pandas as pd
import tensorflow as tf
import open3d as o3d
import numpy as np
import gc



#get google cloud token
import os
os.environ["CLOUDSDK_CONFIG"] = "/home/jacob/.config/gcloud"

import subprocess
token = subprocess.check_output(
    ["/usr/bin/gcloud", "auth", "print-access-token"]
).decode().strip()

from datetime import datetime, timezone, timedelta
fs = pafs.GcsFileSystem(access_token=token, credential_token_expiration=datetime.now(timezone.utc) + timedelta(hours=1))



#to convert spherical coordinates to cartesian

def sphere_to_cart(phi, rho, theta):
    horizontal_d = rho * np.cos(theta)
    X = horizontal_d * np.cos(phi)
    Y = horizontal_d * np.sin(phi)
    Z = rho * np.sin(theta)

    return X, Y, Z






#processing each lidar laser for a given timestamp

def laser_process(laser_num: int, df, df_rgc, timestamp, df_seg=None):

    #df == lidar folder (range image points)
    #df_rgc == calibration folder (angle of each lidar sensor)
    #df_seg == segmentation class labels


    #lidar folder processing
    laser_lidar = df.loc[df["key.laser_name"] == int(laser_num)]
    laser_lidar_t = laser_lidar.loc[laser_lidar["key.frame_timestamp_micros"] == timestamp]
    laser_shape = tuple(laser_lidar_t["[LiDARComponent].range_image_return1.shape"].iloc[0])
    laser_lidar_t_grid = laser_lidar_t["[LiDARComponent].range_image_return1.values"].iloc[0].reshape(laser_shape)

    #calibration folder processing
    laser_calib = df_rgc.loc[df_rgc["key.laser_name"] == int(laser_num)]

    if laser_calib["[LiDARCalibrationComponent].beam_inclination.values"].iloc[0] is not None:
        theta_series = laser_calib["[LiDARCalibrationComponent].beam_inclination.values"].iloc[0]
        theta_series = theta_series[::-1]       #because beam_inclination values are in increasing order, so reverse it for laser 1

    else:
        max_inclin = laser_calib["[LiDARCalibrationComponent].beam_inclination.max"].iloc[0]
        min_inclin = laser_calib["[LiDARCalibrationComponent].beam_inclination.min"].iloc[0]
        theta_series = np.linspace(max_inclin, min_inclin, num=laser_shape[0])
    
    #extrinsic matrix — read here because the azimuth grid needs the sensor's yaw,
    #and reused below for the sensor -> global transform
    ex_transform = np.array(laser_calib["[LiDARCalibrationComponent].extrinsic.transform"].iloc[0]).reshape(4, 4)

    #the range image columns are aligned to the vehicle's forward axis, so we subtract
    #this sensor's mounting yaw (atan2 of the extrinsic rotation) to get true azimuth
    az_correction = np.arctan2(ex_transform[1, 0], ex_transform[0, 0])

    #converting to cartesian from spherical range image
    theta_array = np.array(theta_series)
    phi_array = np.linspace(np.pi, np.pi * -1, num=laser_shape[1]) - az_correction

    #range channel / masking
    range_channel = laser_lidar_t_grid[:, :, 0]
    ranges = range_channel[:, :]
    range_mask = ranges > 0

    # meshgrid(phi_array, theta_array) returns azimuth first, elevation second.
    # phi (azimuth) varies across columns; theta (elevation) varies across rows.
    azimuth_grid, elevation_grid = np.meshgrid(phi_array, theta_array)

    X_unmasked, Y_unmasked, Z_unmasked = sphere_to_cart(azimuth_grid, range_channel, elevation_grid)

    X = X_unmasked[range_mask]
    Y = Y_unmasked[range_mask]
    Z = Z_unmasked[range_mask]
    
    #extrinsic transformation (to make it global relative to the scene instead of the sensor)
    points = np.column_stack((X, Y, Z))
    homo_coords = np.column_stack((points, np.ones(len(X))))

    homo_coords_T = np.transpose(homo_coords)
    global_coords_T = np.dot(ex_transform, homo_coords_T)
    global_coords_1 = np.transpose(global_coords_T)

    #drop 4th column to get the global X, Y, Z coords
    global_coords = global_coords_1[:, :3]


    if df_seg is not None:
        #process segmentation labels 
        laser_seg = df_seg.loc[df_seg["key.laser_name"] == int(laser_num)]
        if not laser_seg.empty:
            laser_seg_t = laser_seg.loc[laser_seg["key.frame_timestamp_micros"] == timestamp]
            if laser_seg_t.empty:
                return global_coords, np.zeros(len(global_coords), dtype=int)
        else: 
            return global_coords, np.zeros(len(global_coords), dtype=int) 
        seg_shape = tuple(laser_seg_t["[LiDARSegmentationLabelComponent].range_image_return1.shape"].iloc[0])
        laser_seg_t_grid = laser_seg_t["[LiDARSegmentationLabelComponent].range_image_return1.values"].iloc[0].reshape(seg_shape)
        
        #mask the seg grid to get the true segmentation labels of each X, Y, Z
        seg_grid_masked = laser_seg_t_grid[range_mask]
        seg_grid_masked_coords = seg_grid_masked[:, 1]

        return global_coords, seg_grid_masked_coords
    
    else:
        return global_coords, None







#creating index mapping of lidar files and timestamps

def folder_file_indexer(folder, start_folder_index, end_folder_index):
    file_timestamp_index = []
    lidar_files = fs.get_file_info(pafs.FileSelector(f"waymo_open_dataset_v_2_0_0/{folder}"))
    for i in range(len(lidar_files[start_folder_index:end_folder_index])):  #only start to end files
        pf_lidar = pq.ParquetFile(lidar_files[i].path, filesystem=fs)
        for j in range(pf_lidar.num_row_groups):

            rg = pf_lidar.read_row_group(j, columns=["key.frame_timestamp_micros"])
            df_idx = rg.to_pandas()

            for stamp in df_idx["key.frame_timestamp_micros"].unique():
                index_file_pair = tuple((lidar_files[i].path, stamp))
                file_timestamp_index.append(index_file_pair)


    return file_timestamp_index


# === USAGE ===
# seg_timestamp_index = folder_file_indexer(folder="training/lidar_segmentation/", start_folder_index=0, end_folder_index=10)
# print(seg_timestamp_index)
# lidar_timestamp_index = folder_file_indexer(folder="training/lidar/", start_folder_index=0, end_folder_index=10)
# print(lidar_timestamp_index)




#dataframe function to fetch row groups with the same timestamp between files (lidar and seg)

def timestamp_aligner(file_path, timestamp):
    file_pq = pq.ParquetFile(file_path, filesystem=fs)
    for i in range(file_pq.num_row_groups):
        rg_time = file_pq.read_row_group(i, columns=["key.frame_timestamp_micros"])
        df_time = rg_time.to_pandas()
        if df_time["key.frame_timestamp_micros"].isin([timestamp]).any():
            rg_all = file_pq.read_row_group(i)
            df_all = rg_all.to_pandas()
            return df_all


# === USAGE ===
# seg_test_file, seg_test_timestamp = seg_timestamp_index[100]
# for tup in lidar_timestamp_index:
#     if seg_test_timestamp in tup:
#         lidar_test_file, lidar_test_timestamp = tup
#         lidar_test_df = timestamp_aligner(lidar_test_file, lidar_test_timestamp)
#         seg_test_df = timestamp_aligner(seg_test_file, seg_test_timestamp)
#         break


# print(lidar_test_df.info())
# print(seg_test_df.info())




#matching row groups of timestamps as a whole between lidar and seg files instead of just timestamps
#allows us to retrieve multiple frames efficiently

def scene_processor(scene_base_path, seg_timestamp_index, lidar_timestamp_index):
    lidar_scene_animation = []

    seg_file_path = os.path.join(os.path.dirname(seg_timestamp_index[0][0]), scene_base_path)
    lidar_file_path = os.path.join(os.path.dirname(lidar_timestamp_index[0][0]), scene_base_path)

    calib_pq = pq.ParquetFile(f"waymo_open_dataset_v_2_0_0/training/lidar_calibration/{scene_base_path}", filesystem=fs)
    calib_df = calib_pq.read_row_group(0).to_pandas()

    seg_pq = pq.ParquetFile(seg_file_path, filesystem=fs)
    lidar_pq = pq.ParquetFile(lidar_file_path, filesystem=fs)

    lidar_df = None
    lidar_df_timestamps = set()

    for i in range(seg_pq.num_row_groups):
        seg_df = seg_pq.read_row_group(i).to_pandas()

        for timestamp in seg_df["key.frame_timestamp_micros"].unique():
            #new lidar row group only when the current one doesn't cover this timestamp
            if timestamp not in lidar_df_timestamps:
                lidar_df = None
                for j in range(lidar_pq.num_row_groups):

                    lidar_times = lidar_pq.read_row_group(j, columns=["key.frame_timestamp_micros"]).to_pandas()
                    if lidar_times["key.frame_timestamp_micros"].isin([timestamp]).any():
                        
                        lidar_df = lidar_pq.read_row_group(j).to_pandas()
                        lidar_df_timestamps = set(lidar_df["key.frame_timestamp_micros"].unique())
                        break

            if lidar_df is None:
                continue

            g_coords_1, masked_labels_1 = laser_process(laser_num=1, df=lidar_df, df_rgc=calib_df, df_seg=seg_df, timestamp=timestamp)
            g_coords_2, masked_labels_2 = laser_process(laser_num=2, df=lidar_df, df_rgc=calib_df, df_seg=seg_df, timestamp=timestamp)
            g_coords_3, masked_labels_3 = laser_process(laser_num=3, df=lidar_df, df_rgc=calib_df, df_seg=seg_df, timestamp=timestamp)
            g_coords_4, masked_labels_4 = laser_process(laser_num=4, df=lidar_df, df_rgc=calib_df, df_seg=seg_df, timestamp=timestamp)
            g_coords_5, masked_labels_5 = laser_process(laser_num=5, df=lidar_df, df_rgc=calib_df, df_seg=seg_df, timestamp=timestamp)

            g_coords_concat = np.concatenate((g_coords_1, g_coords_2, g_coords_3, g_coords_4, g_coords_5))
            masked_labels_concat = np.concatenate((masked_labels_1, masked_labels_2, masked_labels_3, masked_labels_4, masked_labels_5))

            lidar_scene_animation.append((g_coords_concat, masked_labels_concat))

    return lidar_scene_animation


# === USAGE ===
# seg_anim_file, _ = seg_timestamp_index[80]
# base_anim = os.path.basename(seg_anim_file)
# animation_frame_list = scene_processor(base_anim, seg_timestamp_index, lidar_timestamp_index)
