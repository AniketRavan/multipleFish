# % % Script written by Aniket
# Ravan(asravan2 @ illinois.edu)
#
#  This script evaluates correlation coefficients on
#  the real preprocessed images
#
#  User inputs
#
#  lut_b_tail - string input; path to larval tail lookup table bottom camera
#
#  lut_s_tail - string input; path to larval tail lookup table side camera
#
#  proj_params - string input; path to camera projection parameters
#
#  data_folder - string input; path to folder containing preprocessed real images for evaluation
#
#  experiment - string input; training experiment
#
#  model_prediction_path - string input;
#
#  User outputs
#
#  A mat file containing the correlation coefficients for projections of the bottom camera and the two side cameras
#
#  User inputs
import os
import pdb
import torch
import numpy as np
import math as m
import cv2 as cv
import imageio.v3 as iio
from scipy.io import loadmat
from scipy.io import savemat
from scipy.interpolate import splprep, splev
from fish_rendering_functions import return_graymodels_fish
from f_interparc import interparc

def roundHalfUp(a):
    return (np.floor(a)+ np.round(a - (np.floor(a)-1))-1)

def uint8(a):
    a = roundHalfUp(a)

    if np.ndim(a) == 0:
        if a <0:
            a = 0
        if a > 255:
            a = 255
    else:
        a[a>255]=255
        a[a<0]=0
    return a

def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta), m.cos(theta)]])


def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])


def Rz(theta):
    return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                      [m.sin(theta), m.cos(theta), 0],
                      [0, 0, 1]])

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    pdb.set_trace()
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def mask_real_image(im_real,im_gray):
    #I will make this better soon by storing structuring elements in a file
    diskSe = [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
    diskSe = np.array(diskSe)
    diskSe = diskSe.astype('uint8')
    im_gray = im_gray.astype('uint8')
    thresh, throwaway = cv.threshold(im_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    bw = np.copy(im_gray)
    bw[bw < thresh] = 0
    bw[bw >= thresh] = 1

    mask = cv.dilate(bw, diskSe)
    im_masked = uint8(mask) * im_real
    return im_masked

##########################################################
#Start evaluation
# im: input image, background subtracted and cropped 
# reconstructed_pose: predictions from residual blocks
# crop_b, crop_s1, crop_s2: bounding box coordinates (top, down, left, right)
# proj_params: camera calibration parameters
# lut_b_tail, lut_s_tail: lookup table for tail

def calculate_correlation_coefficient(im, reconstructed_pose, crop_b, crop_s1, crop_s2, proj_params, lut_b_tail, lut_s_tail):
    s = np.linspace(0,1,10)

    if file % 500 == 0:
        print('File '+ str(file) + ' of ' + str(len(filenames_mat)))


    c = coor_3d
    coor_3d_shifted = coor_3d
    coor_3d_shifted[:,0:10] = (interparc(s,coor_3d[0, 0: 10], coor_3d[1, 0: 10], coor_3d[2, 0: 10],'spline')).T

    coor_3d[:, 0: 10] = coor_3d_shifted[:, 0: 10]
    # Convert pose to params

    subtractionColumn = np.copy(coor_3d_shifted[:, 0])
    subtractionMatrix = np.tile((subtractionColumn).reshape((-1,1)),(1,coor_3d_shifted.shape[1]))
    coor_3d_shifted = coor_3d_shifted - subtractionMatrix 

    diff_vec = coor_3d_shifted[:, 1: 10] - coor_3d_shifted[:, 0: 9]

    heading_vec = diff_vec[:, 0]
    heading_vec_2d = heading_vec[0:2]

    theta_0 = np.arctan2(heading_vec_2d[1], heading_vec_2d[0])


    if (theta_0 < 0):
        theta_0 = 2 * np.pi - np.abs(theta_0)


    phi_0 = -np.arctan2(heading_vec[2], np.linalg.norm(heading_vec_2d))

    coor_3d_fish_frame = Ry(-phi_0) * Rz(-theta_0) * coor_3d_shifted

    eye_vec = coor_3d_fish_frame[:, 10] - coor_3d_fish_frame[:, 11]

    gamma_01 = np.arctan2(eye_vec[2], eye_vec[1])

    ref_vec_for_roll = Rz(gamma_01 + np.pi) * np.array([[1], [0], [0]])

    gamma_02 = np.arctan2(ref_vec_for_roll[1], ref_vec_for_roll[0])

    coor_3d_fish_frame_1 = Rx(-gamma_01) * coor_3d_fish_frame
    coor_3d_fish_frame_2 = Rx(-gamma_02) * coor_3d_fish_frame

    if (np.abs(gamma_01) < np.abs(gamma_02)):
        gamma_0 = gamma_01
        coor_3d_fish_frame = coor_3d_fish_frame_1

    else:
        gamma_0 = gamma_02
        coor_3d_fish_frame = coor_3d_fish_frame_2


    diff_vec_fish_frame = coor_3d_fish_frame[:, 1: 10] - coor_3d_fish_frame[:, 0: 9]


    theta = np.zeros(8)
    phi = np.zeros(8)

    for i in range(0,8):
        theta[i] = np.arctan2(diff_vec_fish_frame[1, i + 1], diff_vec_fish_frame[0, i + 1])
        phi[i] = -np.arctan2(diff_vec_fish_frame[2, i + 1], np.linalg.norm(diff_vec_fish_frame[0:2, i + 1]))

    x = np.zeros(22)
    x[0:3] = coor_3d[:, 2]
    x[3] = theta_0
    x[12] = phi_0
    x[21] = gamma_0
    x[4:12] = np.diff(np.concatenate(([0], theta)))
    x[13:21] = np.diff(np.concatenate(([0], phi)))

    seglen = np.linalg.norm(diff_vec[:, 0])

    #Rendering the images
    fishlen = seglen / 0.09

    imageSizeX = 141
    imageSizeY = 141

    ######Also changed this line to accomodate the tensor ######
    Crop_b = (crop_coor[0,0:4]).numpy() + 1
    Crop_s1 = (crop_coor[0,4:8]).numpy() + 1
    Crop_s2 = (crop_coor[0,8:12]).numpy() + 1


    [gray_b, gray_s1, gray_s2, c_b, c_s1, c_s2, eye_b, eye_s1, eye_s2, coor_3d] = \
        return_graymodels_fish(x, Crop_b, Crop_s1, Crop_s2,
        proj_params, fishlen, imageSizeX, imageSizeY)

    im_b = im[0:141,:]
    im_s1 = im[141:282,:]
    im_s2 = im[282:,:]

    im_b_masked = mask_real_image(im_b, gray_b)
    im_s1_masked = mask_real_image(im_s1, gray_s1)
    im_s2_masked = mask_real_image(im_s2, gray_s2)

    pb = np.corrcoef(im_b_masked[im_b_masked > 0], gray_b[im_b_masked > 0])[0,1]
    ps1 = np.corrcoef(im_s1_masked[im_s1_masked > 0], gray_s1[im_s1_masked > 0])[0,1]
    ps2 = np.corrcoef(im_s2_masked[im_s2_masked > 0], gray_s2[im_s2_masked > 0])[0,1]
    rgb_b = np.zeros( np.concatenate((list(im_b.shape), [3])) )

    #rgb_b[:,:, 2] = uint8(im_b - gray_b)
    #rgb_b[:,:, 1] = uint8(gray_b - im_b)


    #rgb_s1 = np.zeros( np.concatenate((list(im_s1_masked.shape), [3])) )
    #rgb_s1[:,:, 2] = uint8(im_s1 - gray_s1)
    #rgb_s1[:,:, 1] = uint8(gray_s1 - im_s1)

    #rgb_s2 = np.zeros( np.concatenate((list(im_s2_masked.shape), [3])) )
    #rgb_s2[:,:, 2] = uint8(im_s2 - gray_s2)
    #rgb_s2[:,:, 1] = uint8(gray_s2 - im_s2)

    #rgb_im = np.concatenate((rgb_b, rgb_s1, rgb_s2),axis=0)
    return pb, ps1, ps2

    #Save outputs

    savemat('../generate_manuscript_figures_1/data_files_for_figures/correlation_coefficients/correlation_coefficients_'+ \
      experiment + '.mat', {'pb':pb, 'ps1':ps1, 'ps2':ps2})
