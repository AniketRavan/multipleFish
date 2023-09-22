import cv2
import numpy as np
import scipy.io as sio
from pathlib import Path
import os
import pdb
import logging
from pose_prediction_helper_functions import *

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


#try:
for _ in range(0,1):
    #for videoFilePath, dataTime in zip(videoFilePaths, dataTimeArray):
    for _ in range(0, 1):

        logging.info(f"Analyzing video file: {dataTimeArray[0]}")
        #videoFileName = dataDate + '/' + dataTime
        videoFileName = dataDate + '/' + dataTimeArray[0]
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
        print('k0')
        # Define output folder for pose predictions data
        dataForEvaluationParentFolder = 'outputs/data_for_eval_pose_predictions/'
        dataForEvaluationFolder = dataForEvaluationParentFolder + videoFileName + '/'
        path = Path(dataForEvaluationFolder)
        path.mkdir(parents=True, exist_ok=True)
        # Load video
        rgbImage = readVideo(videoFilePaths[0])
        fileMat = sio.loadmat('/home/asravan2/YOLO_and_resnet/outputs/images_with_YOLO_and_resnet/101419/1347/tracked_fish.mat')
        allFish = fileMat['all_fish']
        nFish = allFish.shape[1]
        for frame in range(0, rgbImage.shape[0]):
            _, axs = plt.subplots(nrows=1, ncols=3)
            for fishIdx in range(nFish):
                axs[0].imshow(rgbImage[frame, :, :, 0], cmap='gray')
                axs[0].scatter(allFish[0, fishIdx]['pose_b'][0, 0][frame, 0, :], allFish[0, fishIdx]['pose_b'][0, 0][frame, 1, :], s=0.01, c='green', alpha=0.6)
                axs[0].axis('off')
                axs[1].imshow(rgbImage[frame, :, :, 1], cmap='gray')
                axs[1].scatter(allFish[0, fishIdx]['pose_s1'][0, 0][frame, 0, :], allFish[0, fishIdx]['pose_s1'][0, 0][frame, 1, :], s=0.01, c='green', alpha=0.6)
                axs[1].axis('off')
                axs[2].imshow(rgbImage[frame, :, :, 2], cmap='gray')
                axs[2].scatter(allFish[0, fishIdx]['pose_s2'][0, 0][frame, 0, :], allFish[0, fishIdx]['pose_s2'][0, 0][frame, 1, :], s=0.01, c='green', alpha=0.6)
                axs[2].axis('off')
            plt.savefig('test_predictions/frame' + str(frame) + '.png')
            plt.close()
#except:
#    pass
