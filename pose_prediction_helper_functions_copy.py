from PIL import Image
import multiprocessing as mp
from triangulation_3d import triangulate_3d
from triangulate_backbone import triangulate_3d_parallel
import torch.nn as nn
import itertools 
import numpy as np
import torch
import matplotlib.pyplot as plt
from avi_r import AVIReader
import pdb
import time
# Make sure to install imageio-ffmpeg using pip
import imageio

## model: Trained YOLO model
## rgbImage: Three-channel image representing the three camera views
## proj_params: Camera calibration parameters

def bgsub(video_mat):
    nFrames = video_mat.shape[0]
    nSampFrame = np.minimum(int(nFrames / 2), 100)
    sampFrame = video_mat[np.linspace(0, nFrames - 1, nSampFrame).astype(int), :, :]
    distinctFrames_sorted = np.sort(sampFrame, axis=0)
    backgroundFrame = distinctFrames_sorted[int(nSampFrame * 0.9), :, :]
    backgroundFrame = backgroundFrame[None, :, :]
    backgroundFrame = np.repeat(backgroundFrame, nFrames, axis=0)
    bgsub_video = np.zeros((nFrames, 488, 648), dtype=np.uint8)
    backgroundFrame[backgroundFrame < video_mat] = video_mat[backgroundFrame < video_mat]
    bgsub_video = backgroundFrame - video_mat
    #for i in range(nFrames):
    #    bgsub_video[i, :, :] = backgroundFrame - video_mat[i, :, :]
    return bgsub_video

def readVideo(videoFileName):
    nFrames = 2000
    vid_b = imageio.get_reader(videoFileName + '_b.avi', format='FFMPEG')
    vid_s1 = imageio.get_reader(videoFileName + '_s1.avi', format='FFMPEG')
    vid_s2 = imageio.get_reader(videoFileName + '_s2.avi', format='FFMPEG')
    rgbImage = np.zeros((nFrames, 488, 648, 3), dtype=np.uint8)
    for i in range(nFrames):
        rgbImage[i,:,:,0] = vid_b.get_data(i)[:,:,0]
        rgbImage[i,:,:,1] = vid_s1.get_data(i)[:,:,0]
        rgbImage[i,:,:,2] = vid_s2.get_data(i)[:,:,0]

    return rgbImage

def calc_proj_w_refra(coor_3d, proj_params):
    fa1p00 = proj_params[0,0]
    fa1p10 = proj_params[0,1]
    fa1p01 = proj_params[0,2]
    fa1p20 = proj_params[0,3]
    fa1p11 = proj_params[0,4]
    fa1p30 = proj_params[0,5]
    fa1p21 = proj_params[0,6]
    fa2p00 = proj_params[1,0]
    fa2p10 = proj_params[1,1]
    fa2p01 = proj_params[1,2]
    fa2p20 = proj_params[1,3]
    fa2p11 = proj_params[1,4]
    fa2p30 = proj_params[1,5]
    fa2p21 = proj_params[1,6]
    fb1p00 = proj_params[2,0]
    fb1p10 = proj_params[2,1]
    fb1p01 = proj_params[2,2]
    fb1p20 = proj_params[2,3]
    fb1p11 = proj_params[2,4]
    fb1p30 = proj_params[2,5]
    fb1p21 = proj_params[2,6]
    fb2p00 = proj_params[3,0]
    fb2p10 = proj_params[3,1]
    fb2p01 = proj_params[3,2]
    fb2p20 = proj_params[3,3]
    fb2p11 = proj_params[3,4]
    fb2p30 = proj_params[3,5]
    fb2p21 = proj_params[3,6]
    fc1p00 = proj_params[4,0]
    fc1p10 = proj_params[4,1]
    fc1p01 = proj_params[4,2]
    fc1p20 = proj_params[4,3]
    fc1p11 = proj_params[4,4]
    fc1p30 = proj_params[4,5]
    fc1p21 = proj_params[4,6]
    fc2p00 = proj_params[5,0]
    fc2p10 = proj_params[5,1]
    fc2p01 = proj_params[5,2]
    fc2p20 = proj_params[5,3]
    fc2p11 = proj_params[5,4]
    fc2p30 = proj_params[5,5]
    fc2p21 = proj_params[5,6]
    npts = coor_3d.shape[2]
    coor_b = np.zeros((coor_3d.shape[0],2,npts))
    coor_s1 = np.zeros((coor_3d.shape[0],2,npts))
    coor_s2 = np.zeros((coor_3d.shape[0],2,npts))
    coor_b[:,0,:] = fa1p00+fa1p10*coor_3d[:,2,:]+fa1p01*coor_3d[:,0,:]+fa1p20*coor_3d[:,2,:]**2+fa1p11*coor_3d[:,2,:]*coor_3d[:,0,:]+fa1p30*coor_3d[:,2,:]**3+fa1p21*coor_3d[:,2,:]**2*coor_3d[:,0,:]
    coor_b[:,1,:] = fa2p00+fa2p10*coor_3d[:,2,:]+fa2p01*coor_3d[:,1,:]+fa2p20*coor_3d[:,2,:]**2+fa2p11*coor_3d[:,2,:]*coor_3d[:,1,:]+fa2p30*coor_3d[:,2,:]**3+fa2p21*coor_3d[:,2,:]**2*coor_3d[:,1,:]
    coor_s1[:,0,:] = fb1p00+fb1p10*coor_3d[:,0,:]+fb1p01*coor_3d[:,1,:]+fb1p20*coor_3d[:,0,:]**2+fb1p11*coor_3d[:,0,:]*coor_3d[:,1,:]+fb1p30*coor_3d[:,0,:]**3+fb1p21*coor_3d[:,0,:]**2*coor_3d[:,1,:]
    coor_s1[:,1,:] = fb2p00+fb2p10*coor_3d[:,0,:]+fb2p01*coor_3d[:,2,:]+fb2p20*coor_3d[:,0,:]**2+fb2p11*coor_3d[:,0,:]*coor_3d[:,2,:]+fb2p30*coor_3d[:,0,:]**3+fb2p21*coor_3d[:,0,:]**2*coor_3d[:,2,:]
    coor_s2[:,0,:] = fc1p00+fc1p10*coor_3d[:,1,:]+fc1p01*coor_3d[:,0,:]+fc1p20*coor_3d[:,1,:]**2+fc1p11*coor_3d[:,1,:]*coor_3d[:,0,:]+fc1p30*coor_3d[:,1,:]**3+fc1p21*coor_3d[:,1,:]**2*coor_3d[:,0,:]
    coor_s2[:,1,:] = fc2p00+fc2p10*coor_3d[:,1,:]+fc2p01*coor_3d[:,2,:]+fc2p20*coor_3d[:,1,:]**2+fc2p11*coor_3d[:,1,:]*coor_3d[:,2,:]+fc2p30*coor_3d[:,1,:]**3+fc2p21*coor_3d[:,1,:]**2*coor_3d[:,2,:]
    return coor_b - 1, coor_s1 - 1, coor_s2 - 1 # Subtract 1 to abide with MATLAB's indices

def pass_on_to_triangulation(bKey, s1Key, s2Key, proj_params):
    pose_b = np.mean(bKey[10:12, :].copy(), axis=0) + 1
    pose_s1 = np.mean(s1Key[10:12, :].copy(), axis=0) + 1
    pose_s2 = np.mean(s2Key[10:12, :].copy(), axis=0) + 1
    return triangulate_3d(pose_b, pose_s1, pose_s2, proj_params)

def iterateFish_parallel(bKeys, bBoxes, s1Keys, s1Boxes, s2Keys, s2Boxes, triTresh, proj_params, pool):
    bFish = []
    s1Fish = []
    s2Fish = []
    bFishBox = []
    s1FishBox = []
    s2FishBox = []
    # Create an iterable for all combinations of fish indices in the three cameras
    iteratedKeys = list(itertools.product(bKeys, s1Keys, s2Keys, [proj_params]))
    iteratedBoxes = list(itertools.product(bBoxes, s1Boxes, s2Boxes))
    _, lossArray = zip(*pool.starmap(pass_on_to_triangulation, iteratedKeys))
    idx = 0
    for result, iteratedKey, iteratedBox in zip(lossArray, iteratedKeys, iteratedBoxes):
        if (result < triTresh):
            bFish.append(iteratedKey[0])
            bFishBox.append(iteratedBox[0])
            s1Fish.append(iteratedKey[1])
            s1FishBox.append(iteratedBox[1])
            s2Fish.append(iteratedKey[2])
            s2FishBox.append(iteratedBox[2])
        idx += 1
    return bFish, bFishBox, s1Fish, s1FishBox, s2Fish, s2FishBox



def return_YOLO_output(model, rgbImage, proj_params, pool):
    results = model(rgbImage, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    keypoints = results.keypoints.xy.cpu().numpy()

    bBoxes = []
    s1Boxes = []
    s2Boxes = []

    bKeys = []
    s1Keys = []
    s2Keys = []

    # Orginizing YOLO's Output
    idx = 0
    for cls in classes:
        if cls == 0.0 :
            bBoxes.append(boxes[idx,:])
            bKeys.append(keypoints[idx])
        if cls == 1.0 :
            s1Boxes.append(boxes[idx,:])
            s1Keys.append(keypoints[idx])
        if cls == 2.0 :
            s2Boxes.append(boxes[idx,:])
            s2Keys.append(keypoints[idx])
        idx += 1
    fInB = len(bKeys)
    fInS1 = len(s1Keys)
    fInS2 = len(s2Keys)

    # Just a list to see how bad the triangulation was
    triangulationError = [ ]

    # This is an arbitrary threshold, there is probably something better
    triTresh = 1
    bFish, bFishBox, s1Fish, s1FishBox, s2Fish, s2FishBox = iterateFish_parallel(bKeys, bBoxes, s1Keys, s1Boxes, s2Keys, s2Boxes, triTresh, proj_params, pool)
    return bFish, bFishBox, s1Fish, s1FishBox, s2Fish, s2FishBox


def padImage(bIm, bFishBox, s1Im, s1FishBox, s2Im, s2FishBox, newImageX, newImageY):
    nFish = len(bFishBox)
    imResnet = torch.zeros(nFish, 3, newImageY, newImageX)
    listBox = [bFishBox, s1FishBox, s2FishBox]
    viewImages = [bIm, s1Im, s2Im]
    # Array storing 'extra' offset in cropping introduced by padding: nFish, 3 cameras, 2 (side, top)
    padding_crop_extra = np.zeros((nFish, 3, 2))
    for fishIdx in range(nFish):
        for viewIdx in range(3):
            viewBoxes = listBox[viewIdx]
            Box = viewBoxes[fishIdx]
            img = torch.tensor(viewImages[viewIdx])

            # Crop image
            subIm = img[int(Box[1]):int(Box[3]), int(Box[0]):int(Box[2])]

            h_buffer = newImageY - subIm.shape[0]
            w_buffer = newImageX - subIm.shape[1]

            # pad_tuple: (left, right, top, bottom)
            pad_tuple = (int(w_buffer / 2), w_buffer - int(w_buffer / 2), int(h_buffer / 2), h_buffer - int(h_buffer / 2))
            padding = nn.ConstantPad2d(pad_tuple, 0)
            imResnet[fishIdx, viewIdx, :, :] = padding(subIm) / torch.max(subIm)
            padding_crop_extra[fishIdx, viewIdx, 0] = pad_tuple[0]
            padding_crop_extra[fishIdx, viewIdx, 1] = pad_tuple[2]
    return imResnet, padding_crop_extra


def plot_pose_predictions(img, yoloResnetFolder, fileName, p_b, p_s1, p_s2):
    _,axs = plt.subplots(nrows=1, ncols=3)
    axs[0].imshow(img[0, :, :].cpu().detach().numpy(), cmap='gray')
    axs[0].scatter(p_b[0, 0:10], p_b[1, 0:10], c='green', s = 0.7)
    axs[0].scatter(p_b[0, 10:12], p_b[1, 10:12], c='red', s=0.7)


    axs[1].imshow(img[1, :, :].cpu().detach().numpy(), cmap='gray')
    axs[1].scatter(p_s1[0, 0:10], p_s1[1, 0:10], c='green', s = 0.7)
    axs[1].scatter(p_s1[0, 10:12], p_s1[1, 10:12], c='red', s=0.7)

    axs[2].imshow(img[2, :, :].cpu().detach().numpy(), cmap='gray')
    axs[2].scatter(p_s2[0, 0:10], p_s2[1, 0:10], c='green', s = 0.7)
    axs[2].scatter(p_s2[0, 10:12], p_s2[1, 10:12], c='red', s=0.7)


    fileName = yoloResnetFolder + fileName
    plt.savefig(fileName)
    plt.close()
