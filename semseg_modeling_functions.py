import gcsfs
from google.cloud import storage
import pyarrow.parquet as pq
import pyarrow.fs as pafs
import pandas as pd
import tensorflow as tf
import open3d as o3d
import numpy as np
import gc
import os
import random
import subprocess
from datetime import datetime, timezone, timedelta

import semseg_functions

import torch
import torch.nn as nn
import torch.nn.functional as F







#stops token from expering during training (since I am reading from GCS, not disk)
def refresh_gcs():
    token = subprocess.check_output(
        ["/usr/bin/gcloud", "auth", "print-access-token"]
    ).decode().strip()
    new_fs = pafs.GcsFileSystem(
        access_token=token,
        credential_token_expiration=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    globals()["fs"] = new_fs
    semseg_functions.fs = new_fs
    return new_fs

fs = refresh_gcs()



if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")






def frame_loader(cache_dir, basefile, timestamp, sample_size=4096, seed=42):

    calib_pq = pq.ParquetFile(f"waymo_open_dataset_v_2_0_0/training/lidar_calibration/{basefile}", filesystem=fs)
    calib_df = calib_pq.read_row_group(0).to_pandas()

    #load finished arrays in cache if they exist (they most likely will with set random seeding)
    cache_path = os.path.join(cache_dir, f"{basefile}_{timestamp}_{sample_size}_{seed}.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["points"], cached["labels"]

    # seg_pq = pq.ParquetFile(f"waymo_open_dataset_v_2_0_0/training/lidar_segmentation/{basefile}", filesystem=fs)
    # lidar_pq = pq.ParquetFile(f"waymo_open_dataset_v_2_0_0/training/lidar/{basefile}", filesystem=fs)

    lidar_df = semseg_functions.timestamp_aligner(f"waymo_open_dataset_v_2_0_0/training/lidar/{basefile}", timestamp=timestamp)
    seg_df = semseg_functions.timestamp_aligner(f"waymo_open_dataset_v_2_0_0/training/lidar_segmentation/{basefile}", timestamp=timestamp)

    g_coords_1, masked_labels_1 = semseg_functions.laser_process(laser_num=1, df=lidar_df, df_rgc=calib_df, df_seg=seg_df, timestamp=timestamp)

    #sampling, as model expects fixed input size
    rng = np.random.default_rng(seed=seed)
    random_indices = rng.integers(low=0, high=len(g_coords_1), size=sample_size)
    random_points = g_coords_1[random_indices]
    random_labels = masked_labels_1[random_indices]

    #normalizing, keeps points on a consistent scale for easier learning
    points_min = random_points.min(axis=0)
    points_max = random_points.max(axis=0)

    random_points_norm = (random_points - points_min) / (points_max - points_min)

    #casting and saving to file
    random_points_norm = random_points_norm.astype(np.float32)
    random_labels = random_labels.astype(np.int64)
    np.savez(cache_path, points=random_points_norm, labels=random_labels)

    return random_points_norm, random_labels







#loads a batch from of random files from either training or validation sets (after split)
def batch_loader(specified_set, seed=42):
    # random.seed(seed)
    # sample_list = random.sample(specified_set, k=K)

    point_clouds = []
    label_sets = []

    for sample in specified_set:

        file_name, timestamp = sample
        base_file_name = os.path.basename(file_name)
        points, labels = frame_loader(basefile=base_file_name, timestamp=timestamp)

        #scanning the labels and points to see if they are valid instances
        unique_labels, label_counts_by_index = np.unique(labels, return_counts=True)
        #print("Unique labels per point cloud: ", (len(unique_labels)))
        # print("How many times does each label occur: ", label_counts_by_index)

        if len(unique_labels) < 2:
            raise ValueError("not enough labels")
        
        point_clouds.append(points)
        label_sets.append(labels)

    
    # points_labels = np.vstack((point_clouds, label_sets))
    #stack point clouds and labels from each sample into two distinct lists
    stacked_points = np.stack(point_clouds)
    stacked_labels = np.stack(label_sets)

    
    return stacked_points, stacked_labels


    

#splitting the data up on an indexed folder   
def train_val_split(index, train_split=80, seed=42):

    training_file_list = []
    val_file_list = []

    random.seed(seed)
    set_of_files = set()
    
    #extract unique file-basenames
    for file, _ in index:
        base_file = os.path.basename(file)
        set_of_files.add(base_file)

    #shuffle them for random order
    list_of_files = list(set_of_files)
    shuffled_files = random.sample(list_of_files, len(set_of_files))

    #split them up (train and validation folders)
    counter=0
    train_split_percentage = train_split / 100
    for unique_file in shuffled_files:
        if counter < len(shuffled_files) * train_split_percentage:
            training_file_list.append(unique_file)
            counter += 1
        else:
            val_file_list.append(unique_file)

    # return training_file_list, val_file_list 
    # adding the frames to the files after splitting for the dataloader
    training_frames = []
    val_frames = []

    for split_file, timestamp in index:
        base_split_file = os.path.basename(split_file)
        if base_split_file in training_file_list:
            training_frames.append((base_split_file, timestamp))
        elif base_split_file in val_file_list:
            val_frames.append((base_split_file, timestamp))
        else:
            raise ValueError("This basefile in the main index was NOT in the training or test file lists")

    return training_frames, val_frames


    


#testing on seg
# seg_training = semseg_functions.folder_file_indexer(folder="training/lidar_segmentation/", start_folder_index=0, end_folder_index=2)
    
# training, validating = train_val_split(seg_training[0:5])
# points, labels = batch_loader(specified_set=training)

# print(f"Shape of stacked points: {points.shape}")
# print(f"Shape of stacked labels: {labels.shape}")






def calculate_matrix(preds, targets, num_classes, ignore_index=None):
    """
    Calculates the mean Intersection over Union (mIoU).
    Args:
        preds (Tensor): Predicted mask of shape (B, H, W) or flattened.
        targets (Tensor): Ground truth mask of same shape.
        num_classes (int): Total number of semantic classes.
        ignore_index (int, optional): Class index to ignore (e.g., background/void).
    """
    # Flatten tensors to 1D
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    # Filter out the ignore index if specified
    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]

    # Calculate confusion matrix: rows = targets, cols = preds
    # bincount constructs a flat matrix of size num_classes^2
    indices = num_classes * targets + preds
    conf_matrix = torch.bincount(indices, minlength=num_classes**2)
    conf_matrix = conf_matrix.reshape(num_classes, num_classes)

    return conf_matrix

def calculate_iou_miou(conf_matrix, ignore_index=None):

    # Intersection is the diagonal elements (TP)
    intersection = torch.diag(conf_matrix)
    
    # Union = TP + FP + FN
    # Ground truth per class (axis 1) + Predictions per class (axis 0) - Intersection
    union = conf_matrix.sum(dim=1) + conf_matrix.sum(dim=0) - intersection

    # Calculate IoU per class
    # Use a small epsilon to avoid division by zero if a class is entirely absent
    iou = intersection.float() / (union.float() + 1e-10)
    
    # Exclude classes that are not present in either targets or predictions
    present_classes = (union > 0)
    if ignore_index != None:
        present_classes[ignore_index] = False
    if present_classes.sum() == 0:
        return iou, torch.tensor(0.0)
        
    mean_iou = iou[present_classes].mean()
    return iou, mean_iou









#count classes per points for Weighted Cross-Entropy Loss

def class_per_point_weights(frames, num_classes, K=20, ignore_index=None):

    running_class = torch.zeros(23)

    for frame_step in range(0, len(frames), K):

        _, labels = batch_loader(frames[frame_step:frame_step+K])
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        labels_flattened = labels_tensor.contiguous().view(-1)
        class_array = torch.bincount(labels_flattened, minlength=num_classes)

        running_class += class_array

    #ignore the 0 class
    if ignore_index is not None:
        mask = running_class > 0
        mask[ignore_index] = False
    

    #inverse median frequency
    total_counts = running_class.sum().float()
    frequencies = running_class.float() / total_counts
    median_freq = torch.median(frequencies[mask])

    class_weights = torch.zeros(23)
    class_weights[mask] = median_freq / frequencies[mask]

    #normalize
    class_weights[mask] = class_weights[mask] / class_weights[mask].mean()

    return class_weights


        
