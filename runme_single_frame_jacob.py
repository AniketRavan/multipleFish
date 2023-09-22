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

model_file = 'resnet_pose_100_percent.pt'
modelPath = 'best.pt' 
imFolder = 'inputs/frames/'
imName = 'frame_100.png'
name = imName[:-4]
imPath = imFolder + imName

yoloCropsParentFolder = 'outputs/images_cropped_with_YOLO/'
yoloCropsFolder = yoloCropsParentFolder + name + '/'
if os.path.exists(yoloCropsFolder):
    # restart it
    print('hello')
    #su.rmtree(yoloCropsFolder)
    #os.mkdir(yoloCropsFolder)
else:
    os.mkdir(yoloCropsFolder)

yoloResnetParentFolder = 'outputs/images_with_YOLO_and_resnet/'
yoloResnetFolder = yoloResnetParentFolder + name + '/'
if os.path.exists(yoloResnetFolder):
    # restart it
    print('hello')
    #su.rmtree(yoloResnetFolder)
    #os.mkdir(yoloResnetFolder)
else:
    os.mkdir(yoloResnetFolder)

dataForEvaluationParentFolder = 'outputs/data_for_eval_pose_predictions/'
dataForEvaluationFolder = dataForEvaluationParentFolder + name + '/'
if os.path.exists(dataForEvaluationFolder):
    # restart it
    print('hello')
    #su.rmtree(dataForEvaluationFolder)
    #os.mkdir(dataForEvaluationFolder)
else:
    os.mkdir(dataForEvaluationFolder)

#dataForEvaluationCropFolder = dataForEvaluationFolder + 'crop/'
#dataForEvaluationCoorFolder = dataForEvaluationFolder + 'coor_3d/'
#os.mkdir(dataForEvaluationCropFolder)
#os.mkdir(dataForEvaluationCoorFolder)



proj_params = sio.loadmat('proj_params_101019_corrected_new.mat')
proj_params = proj_params['proj_params']
proj_params = proj_params[None, :, :]
# These are the crops assuming that the cameras are pointing at the origin
# (0,0, 70?), and the video frames have a width of 648 and height 488
# this could introduce some error if this is wrong
crop_b = np.array([-12.5, 474.5, -6.5, 640.5])
crop_s1 = np.array([-26.5, 460.5, 12.5, 659.5])
crop_s2 = np.array([-22.5, 464.5, -8.5, 638.5])





model = YOLO(modelPath)

img = Image.open(imPath)

numpydata = asarray(img)

numpydata.shape
bIm = numpydata[0:488, :]
s1Im = numpydata[488: 488 * 2, :]
s2Im = numpydata[488 * 2: 488 * 3, :]


rgbImage = np.zeros((488, 648, 3))
rgbImage[:,:,0] = bIm
rgbImage[:,:,1] = s1Im
rgbImage[:,:,2] = s2Im

pdb.set_trace()
results = model(rgbImage)[0]
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

bFish = []
s1Fish = []
s2Fish = []

bFishBox = []
s1FishBox = []
s2FishBox = []

# Just a list to see how bad the triangulation was
triangulationError = [ ]

# This is an arbitrary threshold, there is probably something better
triTresh = 32

def iterateFishes( bKeys, bBoxes, s1Keys, s1Boxes, s2Keys, s2Boxes):
    fInB = len(bKeys)
    fInS1 = len(s1Keys)
    fInS2 = len(s2Keys)

    for bI in range(fInB):
        for s1I in range(fInS1):
            for s2I in range(fInS2):

                # TODO
                # Modify this part to group fishes better
                # right now it does it by checking the first backbonei point
                pose_b = np.array([ bKeys[bI][0,0], bKeys[bI][0,1] ] )
                pose_s1 = np.array([ s1Keys[s1I][0,0], s1Keys[s1I][0,1] ] )
                pose_s2 = np.array([ s2Keys[s2I][0,0], s2Keys[s2I][0,1] ] )

                pose_b[0] = pose_b[0] + crop_b[2]
                pose_b[1] = pose_b[1] + crop_b[0]
                pose_s1[0] = pose_s1[0] + crop_s1[2]
                pose_s1[1] = pose_s1[1] + crop_s1[0]
                pose_s2[0] = pose_s2[0] + crop_s2[2]
                pose_s2[1] = pose_s2[1] + crop_s2[0]

                x, fun = triangulation_3d_jacob(pose_b, pose_s1, pose_s2, proj_params)
                triangulationError.append(fun)
                print([fun, bI, s1I, s2I])
                if fun < triTresh:
                    return True, bI, s1I, s2I
                                
    return False, None, None, None

while True:
    flag, bI, s1I, s2I = iterateFishes( bKeys, bBoxes, s1Keys, s1Boxes  ,s2Keys, s2Boxes)
    
    if flag == False:
        break
    else:
        bFish.append(bKeys[bI])
        bKeys.pop(bI)
        bFishBox.append(bBoxes[bI])
        bBoxes.pop(bI)
        

        s1Fish.append(s1Keys[s1I])
        s1Keys.pop(s1I)
        s1FishBox.append(s1Boxes[s1I])
        s1Boxes.pop(s1I)
                

        s2Fish.append(s2Keys[s2I])
        s2Keys.pop(s2I)
        s2FishBox.append(s2Boxes[s2I])
        s2Boxes.pop(s2I)


#########################################
newImageY = 141
newImageX = 141
xCenterOfSmallerImage = 70
yCenterOfSmallerImage = 70
amountOfFishes = len(bFish)

lisFish = [bFish, s1Fish, s2Fish]
lisBox = [bFishBox, s1FishBox, s2FishBox]
viewImages = [bIm, s1Im, s2Im]

placeHolder = np.zeros((newImageX, newImageY))

concatList = []
outLis = [placeHolder, placeHolder, placeHolder]

#############################
# variables for plotting back
bMask = np.zeros((488, 648))
s1Mask = np.zeros((488, 648))
s2Mask = np.zeros((488, 648))

resBBoxes = []
resS1Boxes = []
resS2Boxes = []
resBoxes = [resBBoxes, resS1Boxes, resS2Boxes ]

subB = []
subS1 = []
subS2 = []
subContainer = [subB, subS1, subS2]

bPoses = []
s1Poses = []
s2Poses = []
PosesContainer = [bPoses, s1Poses, s2Poses]

############################

# This part gets the fishes specified by the bounding boxes of YOLO
for fishIdx in range(amountOfFishes):
    for viewIdx in range(3):
        viewBoxes = lisBox[viewIdx]
        bBox = viewBoxes[fishIdx] 
        img = viewImages[viewIdx]
        
        resultBoxArr = resBoxes[viewIdx]
        sub = subContainer[viewIdx]

        z = np.zeros((newImageY, newImageX))
        bBox.astype(int)
        hGb = (bBox[1] + bBox[3]) //2
        wGb = (bBox[0] + bBox[2]) //2

        xSub = int(hGb - bBox[1])
        ySub = int(wGb - bBox[0])
        
        
        # TODO
        # Might have to add +1
        subIm = img[ int(bBox[1]):int(bBox[3]), int(bBox[0]):int(bBox[2]) ]
        

        resultBoxArr.append([ int(bBox[1]), int(bBox[0]) ])
        sub.append( [ySub, xSub])

        (subHeight, subWidth) = subIm.shape
        z[ yCenterOfSmallerImage - ySub : (yCenterOfSmallerImage - ySub) + subHeight, \
            xCenterOfSmallerImage - xSub : (xCenterOfSmallerImage - xSub) + subWidth ] = subIm

        outLis[viewIdx] = z
    concatIm = np.concatenate((outLis[0], outLis[1], outLis[2]), axis = 0)
    concatList.append(concatIm)
    cv.imwrite(yoloCropsFolder + 'Fish_' + str(fishIdx) + '.png', concatIm)



# This part gets the resnet output and plots it on the cropped Images
for fishIdx in range(amountOfFishes):
    rgb = np.zeros((3,newImageY, newImageX))

    transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float)])
    picName = 'Fish_' + str(fishIdx) + '.png'
    #rootFolder = paddingOut
    #picName = rootFolder + picName
    picName = yoloCropsFolder + picName
    f = Image.open(picName)
    f = transform(f)

    bImg = f[:,0: 141,:]
    s1Img = f[:,141:2*141,:]
    s2Img = f[:,141*2:141*3,:]
    rgb = torch.zeros(3,141,141)
    rgb[0,...] = bImg
    rgb[1,...] = s1Img
    rgb[2,...] = s2Img
    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet18(3, 12, activation='leaky_relu').to(device)
    #model = resnet18(3, 12, activation='leaky_relu')
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file))
    model.eval()



    with torch.no_grad():
        rgb = rgb.unsqueeze(0)

        rgb.to(device)

        pose_recon_b, pose_recon_s1, pose_recon_s2 = model( rgb )
    pdb.set_trace()
    #image_file = cbook.get_sample_data('im_1.png')
    img = plt.imread(picName)

    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')

    ax.imshow(img, cmap = 'gray')


    p_b = pose_recon_b.cpu().detach().numpy()
    p_b = p_b[0]

    bPoses.append(p_b)

    for pI in range(p_b.shape[1]):
        x , y = p_b[:,pI]
        if pI < 10:
            circ = Circle((x,y),1, color = 'g')
        else:
            circ = Circle((x,y),1, color = 'r')
        ax.add_patch(circ)

    p_s1 = pose_recon_s1.cpu().detach().numpy()
    p_s1 = p_s1[0]

    s1Poses.append(p_s1)

    for pI in range(p_s1.shape[1]):
        x , y = p_s1[:,pI]
        y += 141
        if pI < 10:
            circ = Circle((x,y),1, color = 'g')
        else:
            circ = Circle((x,y),1, color = 'r')
        ax.add_patch(circ)


    p_s2 = pose_recon_s2.cpu().detach().numpy()
    p_s2 = p_s2[0]

    s2Poses.append(p_s2)

    for pI in range(p_s2.shape[1]):
        x , y = p_s2[:,pI]
        y += 141 * 2
        if pI < 10:
            circ = Circle((x,y),1, color = 'g')
        else:
            circ = Circle((x,y),1, color = 'r')
        ax.add_patch(circ)
    
    
    fileName = 'Fish_' + str(fishIdx) + '.jpeg'
    fileName = yoloResnetFolder + fileName
    plt.savefig(fileName)




# This next part plots the YOLO + resnetOutput back 
# aswell as get some variables for the correlation coefficients
resBoxes = [resBBoxes, resS1Boxes, resS2Boxes ]
PosesContainer = [bPoses, s1Poses, s2Poses]
masks = [bMask, s1Mask, s2Mask]

ogImage = cv.imread(imPath)

ogBView = ogImage[:488, ...]
ogS1View = ogImage[488:488*2, ...]
ogS2View = ogImage[488 *2 :, ...]
ogViewsContainer = [ogBView, ogS1View, ogS2View]
biggerImageCropContainer = [crop_b, crop_s1, crop_s2]

crop_b_poseEval = []
crop_s1_poseEval = []
crop_s2_poseEval = []
crop_for_poseEval = [crop_b_poseEval, crop_s1_poseEval, crop_s2_poseEval]

UncroppedBPoses = []
UncroppedS1Poses = []
UncroppedS2Poses = []
poseContainer = [UncroppedBPoses, UncroppedS1Poses, UncroppedS2Poses]
#pdb.set_trace()

for viewIdx in range(3):
    mask = masks[viewIdx]
    viewPoses = PosesContainer[viewIdx]
    boxes = resBoxes[viewIdx]
    ogView = ogViewsContainer[viewIdx]
    subs = subContainer[viewIdx]
    cropContainer = crop_for_poseEval[viewIdx]
    biggerCrop = biggerImageCropContainer[viewIdx]
    poseContainer4View = poseContainer[viewIdx]

    for poseIdx in range(len(viewPoses)):
        y0, x0 = boxes[poseIdx ][0], boxes[ poseIdx][1]
        pose = viewPoses[poseIdx]
        sub = subs[poseIdx]

        
        crop = [biggerCrop[0] + y0 + sub[0] - 70, biggerCrop[0] + y0 + sub[0] - 70 + 140, \
                biggerCrop[2] + x0 + sub[1] - 70, biggerCrop[2] + x0 + sub[1] - 70 + 140]
        #pdb.set_trace()
        cropContainer.append(crop)
        poseWithoutCrop = np.copy(pose)
        poseWithoutCrop[0,:] = pose[0,:] + crop[2]
        poseWithoutCrop[1,:] = pose[1,:] + crop[0]
        poseContainer4View.append( poseWithoutCrop)

        for pIdx in range(12):
            circCoor = (  x0 +  int(pose[0, pIdx]) - 70 + sub[1 ]   , y0 +  int(pose[1, pIdx]) - 70  + sub[0]   )
            
            if pIdx < 10:
                color = (0,255,0) 
            else:
                color = (255,0, 0)

            ogView = cv.circle( ogView, circCoor , 5, color, 1)

triangulated3dCoorsContainer = []
for fishNum in range(len(UncroppedBPoses)):
    triangulateCoors = np.zeros((3,12))
    bPose = UncroppedBPoses[fishNum]
    s1Pose = UncroppedS1Poses[fishNum]
    s2Pose = UncroppedS2Poses[fishNum]
    
    for pIdx in range(12):
        #triangulateCoors[:,pIdx] , loss = triangulation_3d(bPose[:,pIdx], s1Pose[:,pIdx], s2Pose[:,pIdx], proj_params)
        # in accordance to Matlab format
        triangulateCoors[:,pIdx] , loss = triangulation_3d_jacob(bPose[:,pIdx] + [1,1], s1Pose[:,pIdx] + [1,1], s2Pose[:,pIdx] + [1,1], proj_params)
    triangulated3dCoorsContainer.append(np.copy(triangulateCoors))
cropToSave = []
for cIdx in range(len(crop_b_poseEval)):
    b = crop_b_poseEval[cIdx]
    s1 = crop_s1_poseEval[cIdx]
    s2 = crop_s2_poseEval[cIdx]
    catCrop = np.concatenate((b,s1,s2), axis = 0)
    cropToSave.append(catCrop)

triangulated3dCoorsContainer = np.array(triangulated3dCoorsContainer)
cropToSave = np.array(cropToSave)
cropToSave += 1
np.save(dataForEvaluationFolder + 'crop.npy', cropToSave) 
np.save(dataForEvaluationFolder + 'coor.npy', triangulated3dCoorsContainer)





