import ultralytics
import torch
from ultralytics import YOLO
import pdb
from PIL import Image
from numpy import asarray
import numpy as np
import cv2 as cv
import os
from multipleFishFunctionsFile import *
from triangulation_3d import triangulate_3d 
from triangulation_3d_jacob import triangulation_3d_jacob
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from ResNet_Blocks_3D_four_blocks import resnet18
import torch.nn as nn
import torch
import pdb
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cbook as cbook
from PIL import Image
import scipy.io as sio
import shutil as su
from triangulate_backbone import triangulate_3d_parallel, triangulate_3d_return_pose_parallel
from triangulation_3d import triangulate_3d, triangulate_using_rays
import itertools
import multiprocessing as mp
import time
from pose_prediction_helper_functions import *
import cv2
from avi_r import AVIReader
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

#####################################################
# User inputs (input and output file paths)
model_file = 'resnet_pose_100_percent.pt'
modelPath = 'yolo_best_model.pt' 
imFolder = 'inputs/frames/'
videoFolder = 'inputs/videos'
dataPath = '/home/asravan2/YOLO_and_resnet/inputs/'
dataDate = '101419'
lut_a_r_path = 'lut_a_r_101019_corrected_new.mat'
#####################################################

# Find create an array of all video file paths
dataTimeArray = os.listdir(dataPath + dataDate)
videoFilePaths = [dataPath + dataDate + '/' + dataTime + '/' + dataDate + '_' + dataTime for dataTime in dataTimeArray]
lut_a_r = sio.loadmat(lut_a_r_path)['lut_a_r']

pool = mp.Pool(40)
try:
    for videoFilePath, dataTime in zip(videoFilePaths, dataTimeArray):
        logging.info(f"Analyzing video file: {dataTime}")
        videoFileName = dataDate + '/' + dataTime

        # Define output folder for cropped images using bounding box
        yoloCropsParentFolder = 'outputs/images_cropped_with_YOLO/'
        yoloCropsFolder = yoloCropsParentFolder + videoFileName + '/'
        path = Path(yoloCropsFolder)
        path.mkdir(parents=True, exist_ok=True)

        # Define output folder for YOLO + resnet output
        yoloResnetParentFolder = 'outputs/images_with_YOLO_and_resnet/'
        yoloResnetFolder = yoloResnetParentFolder + videoFileName + '/'
        path = Path(yoloResnetFolder)
        path.mkdir(parents=True, exist_ok=True)


        # Define output folder for pose predictions data
        dataForEvaluationParentFolder = 'outputs/data_for_eval_pose_predictions/'
        dataForEvaluationFolder = dataForEvaluationParentFolder + videoFileName + '/'
        path = Path(dataForEvaluationFolder)
        path.mkdir(parents=True, exist_ok=True)


        # Load video
        rgbImage = readVideo(videoFilePath)
        # Load projection parameters
        proj_params = sio.loadmat('proj_params_101019_corrected_new.mat')
        proj_params = proj_params['proj_params']

        # Background subtraction
        bgsubVidB = bgsub(rgbImage[:,:,:,0])
        bgsubVidS1 = bgsub(rgbImage[:,:,:,1])
        bgsubVidS2 = bgsub(rgbImage[:,:,:,2])
        ############################################ 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        yolo_model = YOLO(modelPath)
        res_model = resnet18(3, 12, activation='leaky_relu').to(device)
        res_model = nn.DataParallel(res_model)
        res_model.load_state_dict(torch.load(model_file))
        res_model.eval()

        frame = np.zeros((488, 648, 3))
        

        ############################################
        # Loop over all frames
        nFrames = bgsubVidB.shape[0]
        for i in range(0, nFrames):
            logging.info(f"Analyzing frame {i} of {bgsubVidB.shape[0]}")
            frame[:,:,0] = 255 * (bgsubVidB[i, :, :] / np.max(bgsubVidB[i, :, :]))
            frame[:,:,1] = 255 * (bgsubVidS1[i, :, :] / np.max(bgsubVidS1[i, :, :]))
            frame[:,:,2] = 255 * (bgsubVidS2[i, :, :] / np.max(bgsubVidS2[i, :, :]))
            bFish, bFishBox, s1Fish, s1FishBox, s2Fish, s2FishBox = return_YOLO_output(yolo_model, frame, lut_a_r, pool)
            logging.debug(f"{len(bFish)} fish detected")
            if (len(bFish) == 0):
                continue
            #########################################
            # Pose predictions using residual blocks
            newImageY = 141
            newImageX = 141

            imResnet, padding_offset = padImage(frame[:,:,0], bFishBox, frame[:,:,1], s1FishBox, frame[:,:,2], s2FishBox, newImageX, newImageY)
            nFish = len(bFish)
            with torch.no_grad():
                imResnet.to(device)
                pose_recon_b, pose_recon_s1, pose_recon_s2 = res_model(imResnet)
            pose_recon_b = pose_recon_b.cpu().detach().numpy()
            bFishBox = np.array(bFishBox)
            pose_recon_b = pose_recon_b + bFishBox[:, [0, 1], None] - padding_offset[:, 0, :, None]
            pose_recon_b_backbone = np.reshape(np.swapaxes(pose_recon_b[:, :, 0:10], 0, 1), (2, nFish * 10))
            pose_recon_s1 = pose_recon_s1.cpu().detach().numpy()
            s1FishBox = np.array(s1FishBox)
            pose_recon_s1 = pose_recon_s1 + s1FishBox[:, [0, 1], None] - padding_offset[:, 1, :, None]
            pose_recon_s1_backbone = np.reshape(np.swapaxes(pose_recon_s1[:, :, 0:10], 0, 1), (2, nFish * 10))
            pose_recon_s2 = pose_recon_s2.cpu().detach().numpy()
            s2FishBox = np.array(s2FishBox)
            pose_recon_s2 = pose_recon_s2 + s2FishBox[:, [0, 1], None] - padding_offset[:, 2, :, None]
            pose_recon_s2_backbone = np.reshape(np.swapaxes(pose_recon_s2[:, :, 0:10], 0, 1), (2, nFish * 10))
            reconstructed_3d_backbone, _ = triangulate_3d_parallel(pose_recon_b_backbone.T + 1, pose_recon_s1_backbone.T + 1, pose_recon_s2_backbone.T + 1, proj_params, pool)
            reconstructed_3d_backbone = np.swapaxes(np.reshape(reconstructed_3d_backbone.T, (3, nFish, 10)), 0, 1)

            # Reconstructed 3-D backbone reshaped into nFish x 3 x 10 dimensional array
            reconstructed_3d_backbone = np.reshape(reconstructed_3d_backbone, (nFish, 3, 10))

            # Define for eyes here
            iter_idx_eyes = np.array(list(itertools.product([10, 11], [10, 11], [10, 11])))
            eyes_b_iterated = pose_recon_b[:, :, iter_idx_eyes[:, 0]].copy()
            eyes_s1_iterated = pose_recon_s1[:, :, iter_idx_eyes[:, 1]].copy()
            eyes_s2_iterated = pose_recon_s2[:, :, iter_idx_eyes[:, 2]].copy()
            eyes_b_iterated = np.reshape(np.swapaxes(eyes_b_iterated, 0, 1), (2, nFish * iter_idx_eyes.shape[0]))
            eyes_s1_iterated = np.reshape(np.swapaxes(eyes_s1_iterated, 0, 1), (2, nFish * iter_idx_eyes.shape[0]))
            eyes_s2_iterated = np.reshape(np.swapaxes(eyes_s2_iterated, 0, 1), (2, nFish * iter_idx_eyes.shape[0]))
            eyes_3d, loss_array = triangulate_3d_parallel(eyes_b_iterated.T + 1, eyes_s1_iterated.T + 1, eyes_s2_iterated.T + 1, proj_params, pool)
            eyes_3d = np.swapaxes(np.reshape(eyes_3d.T, (3, nFish, iter_idx_eyes.shape[0])), 0, 1)
            loss_array = np.reshape(loss_array, (nFish, iter_idx_eyes.shape[0]))
            
            # Calculate minimum loss index for each fish
            first_eye_minimum_idx = np.argmin(loss_array[:,:int(iter_idx_eyes.shape[0] / 2)], axis=1)
            eye1_3d = eyes_3d[np.arange(nFish), :, first_eye_minimum_idx].copy()
            eye2_3d = eyes_3d[np.arange(nFish), :, -(first_eye_minimum_idx + 1)].copy() # Negative sign to capture alternate combinations
            pose_3d = np.zeros((nFish, 3, 12))
            pose_3d[:, :, 0:10] = reconstructed_3d_backbone
            pose_3d[:, :, 10] = eye1_3d[:, :].copy() # Don't worry about 0: only one element in this dim
            pose_3d[:, :, 11] = eye2_3d[:, :].copy() # Don't worry about 0: only one element in this dim

            # Calculate pose projection coordinates
            pose_projection_b, pose_projection_s1, pose_projection_s2 = calc_proj_w_refra(pose_3d, proj_params)

            ########################################
            distance_threshold = 2
            if i == 0:
                fish = []
                fishIndices = np.arange(nFish)
                fish_positions_old = np.mean(pose_3d[:, :, 10:12], axis=2)
                for fishIdx in range(nFish):
                    fish.append({})
                    fish[fishIdx]['pose_3d'] = np.empty((nFrames, 3, 12))
                    fish[fishIdx]['pose_b'] = np.empty((nFrames, 2, 12))
                    fish[fishIdx]['pose_s1'] = np.empty((nFrames, 2, 12))
                    fish[fishIdx]['pose_s2'] = np.empty((nFrames, 2, 12))
                    fish[fishIdx]['pose_3d'][i, :, :] = pose_3d[fishIdx, :, :]
                    fish[fishIdx]['pose_b'][i, :, :] = pose_projection_b[fishIdx, :, :]
                    fish[fishIdx]['pose_s1'][i, :, :] = pose_projection_s1[fishIdx, :, :]
                    fish[fishIdx]['pose_s2'][i, :, :] = pose_projection_s2[fishIdx, :, :]
                # Keeping track of unique fish occuring in the video
                maxFishIdx = nFish - 1

            #p_b = pose_projection_b - bFishBox[:, [0, 1], None] + padding_offset[:, 0, :, None]
            #p_s1 = pose_projection_s1 - s1FishBox[:, [0, 1], None] + padding_offset[:, 1, :, None]
            #p_s2 = pose_projection_s2 - s2FishBox[:, [0, 1], None] + padding_offset[:, 2, :, None]
            
            # Plotting pose predictions
            else:
                fish_positions_temp = fish_positions_old
                fish_positions_current = np.mean(pose_3d[:, :, 10:12], axis=2)
                for k, fish_position in enumerate(fish_positions_current):
                    distance_array = np.linalg.norm(fish_positions_old - fish_position, axis=1)
                    min_distance_idx = np.argmin(distance_array)
                    min_distance = distance_array[min_distance_idx]
                    if (min_distance < distance_threshold):
                        fishIndex = fishIndices[min_distance_idx]
                        fish_positions_old[min_distance_idx, :] = fish_positions_current[k, :]
                        fish[fishIndex]['pose_3d'][i, :, :] = pose_3d[k, :, :]
                        fish[fishIndex]['pose_b'][i, :, :] = pose_projection_b[k, :, :]
                        fish[fishIndex]['pose_s1'][i, :, :] = pose_projection_s1[k, :, :]
                        fish[fishIndex]['pose_s2'][i, :, :] = pose_projection_s2[k, :, :]
                    else:
                        maxFishIdx += 1
                        fish.append({})
                        fish[maxFishIdx]['pose_3d'] = np.empty((nFrames, 3, 12))
                        fish[maxFishIdx]['pose_b'] = np.empty((nFrames, 2, 12))
                        fish[maxFishIdx]['pose_s1'] = np.empty((nFrames, 2, 12))
                        fish[maxFishIdx]['pose_s2'] = np.empty((nFrames, 2, 12))
                        fish[maxFishIdx]['pose_3d'][i, :, :] = pose_3d[k, :, :]
                        fish[maxFishIdx]['pose_b'][i, :, :] = pose_projection_b[k, :, :]
                        fish[maxFishIdx]['pose_s1'][i, :, :] = pose_projection_s1[k, :, :]
                        fish[maxFishIdx]['pose_s2'][i, :, :] = pose_projection_s2[k, :, :]
                        fishIndices = np.append(fishIndices, maxFishIdx)
                        fish_positions_temp = np.append(fish_positions_temp, fish_positions_current[None, k, :], axis=0)
                        
                fish_positions_old = fish_positions_temp
                fileName = 'Fish_' + str(fishIdx) + '_' + str(i).rjust(3, '0') + '.png'
                #plot_pose_predictions(img, yoloResnetFolder, fileName, p_b, p_s1, p_s2)
        tracked_fish = {}
        tracked_fish['all_fish'] = fish
        sio.savemat(yoloResnetFolder + 'tracked_fish.mat' , tracked_fish)
    pool.close()

except:
    pass
