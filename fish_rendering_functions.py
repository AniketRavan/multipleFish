import numpy as np
from scipy.io import loadmat
import math as m
import pdb

def roundHalfUp(a):
    return (np.floor(a)+ np.round(a - (np.floor(a)-1))-1)

def mat2gray(imgArg):
    img = np.copy(imgArg)
    minimum = np.min(img)
    maximum = np.max(img)
    difference = maximum - minimum
    
    img = img - minimum
    img = img / difference
    return img

def sigmoid(xArg,scaling):
    x = np.copy(xArg)
    y = 1 / (1 + np.exp(-scaling * x))
    return y

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

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(x-float(mean))**2/(2*var))
    return num/denom

def ind2sub(arr):
    ys = []
    xs = []
    sz = arr.shape
    for x in range(0,sz[0]):
        for y in range(0,sz[1]):
            if arr[y,x] > 0:
                ys.append(y)
                xs.append(x)
    return np.array(ys), np.array(xs)

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

def linIndxTo2DIndx(num, arrShape):
    rows = arrShape[0]

    columnIndx = num / rows
    if np.floor(columnIndx) == columnIndx and columnIndx != 0:
        columnIndx = columnIndx - 1
    else:
        columnIndx = np.floor(columnIndx)

    rowIndx = (num % rows) - 1

    if rowIndx == -1:
        rowIndx = rows - 1

    return (int(rowIndx),int(columnIndx))


    
    
    
def bellymodel(x, y, z, seglen, theta1, phi1, brightness, size_lut):

    #belly_w = seglen * (0.3527); %0.5030;5527
    belly_w = seglen * 0.45
    belly_l = seglen * 1.3000 #1.5970;
    #belly_h = seglen * 0.6531 #0.6294; % 0.55
    #belly_h = seglen * (0.7431);
    belly_h = seglen * 0.6631
    c_belly_wrt_1 = 1-0.475 # 1.0541;

    pt_original = np.zeros((3,3))
    pt_original[:,1] = [size_lut/2, size_lut/2, size_lut/2]
    pt_original[:,0] = pt_original[:,1] - [seglen, 0, 0]
    pt_original[:,2] = pt_original[:,1] + [seglen, 0, 0] #This should not be rotated
    # The belly is rotated twice in reorient_belly_model instead

    belly_c = [c_belly_wrt_1*pt_original[0,0] + (1-c_belly_wrt_1)*pt_original[0,2], c_belly_wrt_1*pt_original[1,0] +
               (1-c_belly_wrt_1)*pt_original[1,2], pt_original[2,0] - seglen/6.5] #3.5602]; % Changed from 6   7.0257

    XX = x - belly_c[0]
    YY = y - belly_c[1]
    ZZ = z - belly_c[2]


    belly_model = np.exp(-2*(XX*XX/(2*belly_l**2) + YY*YY/(2*belly_w**2) +
                             ZZ*ZZ/(2*belly_h**2) - 1))
    belly_model = belly_model*brightness
    return belly_model, belly_c

def calc_proj_w_refra_cpu(coor_3d,proj_params):

    mat = loadmat(proj_params)
    proj_params_mat = mat['proj_params']

    fa1p00 = proj_params_mat[0,0]
    fa1p10 = proj_params_mat[0,1]
    fa1p01 = proj_params_mat[0,2]
    fa1p20 = proj_params_mat[0,3]
    fa1p11 = proj_params_mat[0,4]
    fa1p30 = proj_params_mat[0,5]
    fa1p21 = proj_params_mat[0,6]
    fa2p00 = proj_params_mat[1,0]
    fa2p10 = proj_params_mat[1,1]
    fa2p01 = proj_params_mat[1,2]
    fa2p20 = proj_params_mat[1,3]
    fa2p11 = proj_params_mat[1,4]
    fa2p30 = proj_params_mat[1,5]
    fa2p21 = proj_params_mat[1,6]
    fb1p00 = proj_params_mat[2,0]
    fb1p10 = proj_params_mat[2,1]
    fb1p01 = proj_params_mat[2,2]
    fb1p20 = proj_params_mat[2,3]
    fb1p11 = proj_params_mat[2,4]
    fb1p30 = proj_params_mat[2,5]
    fb1p21 = proj_params_mat[2,6]
    fb2p00 = proj_params_mat[3,0]
    fb2p10 = proj_params_mat[3,1]
    fb2p01 = proj_params_mat[3,2]
    fb2p20 = proj_params_mat[3,3]
    fb2p11 = proj_params_mat[3,4]
    fb2p30 = proj_params_mat[3,5]
    fb2p21 = proj_params_mat[3,6]
    fc1p00 = proj_params_mat[4,0]
    fc1p10 = proj_params_mat[4,1]
    fc1p01 = proj_params_mat[4,2]
    fc1p20 = proj_params_mat[4,3]
    fc1p11 = proj_params_mat[4,4]
    fc1p30 = proj_params_mat[4,5]
    fc1p21 = proj_params_mat[4,6]
    fc2p00 = proj_params_mat[5,0]
    fc2p10 = proj_params_mat[5,1]
    fc2p01 = proj_params_mat[5,2]
    fc2p20 = proj_params_mat[5,3]
    fc2p11 = proj_params_mat[5,4]
    fc2p30 = proj_params_mat[5,5]
    fc2p21 = proj_params_mat[5,6]

    npts = coor_3d.shape[1]

    coor_b = np.zeros((2,npts))
    coor_s1 = np.zeros((2, npts))
    coor_s2 = np.zeros((2, npts))

    coor_b[0,:] = fa1p00 + fa1p10*coor_3d[2,:] + fa1p01*coor_3d[0,:] + fa1p20*(coor_3d[2,:]**2) + fa1p11*coor_3d[2,:]*coor_3d[0,:] + fa1p30*(coor_3d[2,:]**3) + fa1p21*(coor_3d[2,:]**2)*coor_3d[0,:]
    coor_b[1,:] = fa2p00 + fa2p10*coor_3d[2,:] + fa2p01*coor_3d[1,:] + fa2p20*(coor_3d[2,:]**2) + fa2p11*coor_3d[2,:]*coor_3d[1,:] + fa2p30*(coor_3d[2,:]**3) + fa2p21*(coor_3d[2,:]**2)*coor_3d[1,:]
    coor_s1[0,:] = fb1p00 + fb1p10*coor_3d[0,:] + fb1p01*coor_3d[1,:] + fb1p20*(coor_3d[0,:]**2) + fb1p11*coor_3d[0,:]*coor_3d[1,:] + fb1p30*(coor_3d[0,:]**3) + fb1p21*(coor_3d[0,:]**2)*coor_3d[1,:]
    coor_s1[1,:] = fb2p00 + fb2p10*coor_3d[0,:] + fb2p01*coor_3d[2,:] + fb2p20*(coor_3d[0,:]**2) + fb2p11*coor_3d[0,:]*coor_3d[2,:] + fb2p30*(coor_3d[0,:]**3) + fb2p21*(coor_3d[0,:]**2)*coor_3d[2,:]
    coor_s2[0,:] = fc1p00 + fc1p10*coor_3d[1,:] + fc1p01*coor_3d[0,:] + fc1p20*(coor_3d[1,:]**2) + fc1p11*coor_3d[1,:]*coor_3d[0,:] + fc1p30*(coor_3d[1,:]**3) + fc1p21*(coor_3d[1,:]**2)*coor_3d[0,:]
    coor_s2[1,:] = fc2p00 + fc2p10*coor_3d[1,:] + fc2p01*coor_3d[2,:] + fc2p20*(coor_3d[1,:]**2) + fc2p11*coor_3d[1,:]*coor_3d[2,:] + fc2p30*(coor_3d[1,:]**3) + fc2p21*(coor_3d[1,:]**2)*coor_3d[2,:]

    return coor_b, coor_s1, coor_s2

def eye1model(x, y, z, seglen, theta1, phi1, brightness, size_lut, rnd):
    d_eye = seglen * (0.8356)# 1.4332; %0.83
    c_eyes = 1.5230 #1.3015;
    eye1_w = seglen * 0.2097 #0.1671; % 0.35
    eye1_w = seglen * 0.2597
    eye1_w = seglen * 0.2997
    eye1_l = seglen * 0.3306 #0.2507; % 0.45 3006
    eye1_l = seglen * 0.4306
    eye1_h = seglen * 0.2496 #0.2661; % 0.35
    eye1_h = seglen * 0.3096

    pt_original = np.empty([3,3])

    pt_original[:, 1] = np.array([size_lut / 2, size_lut / 2, size_lut / 2])
    pt_original[:, 0] = np.array(pt_original[:, 1] - [seglen, 0, 0])
    temp = Rz(theta1) @ Ry(-phi1) @ np.array([[seglen], [0], [0]])
    pt_original[:, 2] = pt_original[:,1] + np.array([temp[0,0],temp[1,0],temp[2,0]])

    eye1_c = [c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1], c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] + d_eye / 2, pt_original[2, 1] - seglen / 6.3049]


    XX = x - eye1_c[0]
    YY = y - eye1_c[1]
    ZZ = z - eye1_c[2]

    eye1_model = np.exp(-1.2*(XX*XX/(2*eye1_l ** 2) + YY*YY/(2*eye1_w**2) + ZZ*ZZ/(2*eye1_h**2) - 1))
    eye1_model = eye1_model*brightness

    return eye1_model, eye1_c

def eye2model(x, y, z, seglen, theta1, phi1, brightness, size_lut, rnd):
    d_eye = seglen * (0.8356 + (rnd[0] - 0.5)*0.05) # 1.4332; %0.83
    c_eyes = 1.5230 #1.3015;
    eye2_w = seglen * 0.2097 #0.1671; % 0.35
    eye2_w = seglen * 0.2597
    eye2_w = seglen * 0.2997
    eye2_l = seglen * 0.3306 #0.2507; % 0.45
    eye2_l = seglen * 0.4306
    eye2_h = seglen * 0.2496 #0.2661; % 0.35
    eye2_h = seglen * 0.3096

    pt_original = np.empty((3,3))
    pt_original[:,1] = [size_lut/2, size_lut/2, size_lut/2]
    pt_original[:,0] = pt_original[:,1] - [seglen, 0, 0]
    temp = Rz(theta1) * Ry(-phi1)* np.array([[seglen], [0], [0]])
    pt_original[:,2] = pt_original[:,1] + np.array([temp[0,0],temp[1,0],temp[2,0]])

    eye2_c = [c_eyes*pt_original[0,0] + (1-c_eyes)*pt_original[0,1], c_eyes*pt_original[1,0] +
              (1-c_eyes)*pt_original[1,1] - d_eye/2, pt_original[2,1] - seglen/6.3049]


    XX = x - eye2_c[0]
    YY = y - eye2_c[1]
    ZZ = z - eye2_c[2]

    eye2_model = np.exp(-1.2*(XX*XX/(2*eye2_l**2) + YY*YY/(2*eye2_w**2) +
                              ZZ*ZZ/(2*eye2_h**2) - 1))
    eye2_model = eye2_model*brightness


    return eye2_model, eye2_c

def gen_lut_b_tail(n, nseglen, d1, d2, a):
    size_lut = 19
    size_half = (size_lut+1)/2


    random_number = .6
    ballsize = random_number * np.array([3,2,2,2,2,1.5,1.2,1.2,1])
    thickness = random_number * np.array([7,6,5.5,5,4.5,4,3.5,3])
    b_tail = [0.7,0.55,0.45,0.40,0.32,0.28,0.2,0.15]


    imageSizeX = size_lut
    imageSizeY = size_lut
    x = np.linspace(1,imageSizeX,imageSizeX)
    y = np.linspace(1,imageSizeY,imageSizeY)
    [columnsInImage0, rowsInImage0] = np.meshgrid(x, y)


    imblank = np.zeros((size_lut,size_lut))


    radius = ballsize[n]
    th = thickness[n]
    bt = b_tail[n-1]
    bt_gradient = b_tail[n]/b_tail[n-1]

    # p_max = scipy.stats.norm.pdf(0,loc= 0,scale= th)
    p_max = normpdf(0,0,th)

    seglen = 5 + 0.2 * nseglen
    centerX = size_half + d1/5
    centerY = size_half + d2/5

    columnsInImage = columnsInImage0
    rowsInImage = rowsInImage0

    ballpix = (rowsInImage - centerY)**2 + (columnsInImage - centerX)**2 <= radius**2

    # Following lines are equivalent to:
    # ballpix = uint8(ballpix); ballpix = ballpix * 255 * bt * .85
    ballpix = uint8(ballpix)

    ballpix = ballpix * 255
    ballpix = uint8(ballpix)

    ballpix = ballpix * bt
    ballpix = uint8(ballpix)

    ballpix = ballpix * 0.85
    ballpix = uint8(ballpix)


    t = 2*np.pi*(a-1)/360

    pt = np.zeros((2,2))

    R = [[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]

    vec = np.matmul(R,np.array([[seglen] ,[0]]))

    pt[:,0] = np.array([size_half + d1/5, size_half + d2/5])
    pt[:,1] = pt[:,0] + vec[:,0]

    stickpix = imblank

    columnsInImage = columnsInImage0

    rowsInImage = rowsInImage0

    if (pt[0,1] - pt[0,0]) != 0:

        slope = (pt[1,1] - pt[1,0])/(pt[0,1] - pt[0,0])

        #vectors perpendicular to the line segment

        #th is the thickness of the sticks in the model

        vp = np.array([[-slope],[1]]) / np.linalg.norm(np.array([[-slope],[1]]))

        # one vertex of the rectangle

        V1 = pt[:,1] - vp[:,0] * th


        #two sides of the rectangle

        s1 = 2 * vp * th

        s2 = pt[:,0] - pt[:,1]

        # find the pixels inside the rectangle

        r1 = rowsInImage - V1[1]

        c1 = columnsInImage - V1[0]

        #inner products

        ip1 = r1 * s1[1] + c1 * s1[0]

        ip2 = r1 * s2[1] + c1 * s2[0]


        stickpix_bw = (ip1 > 0) * (ip1 < np.dot(s1[:,0], s1[:,0])) * (ip2 > 0) * (ip2 < np.dot(s2,s2))

    else:
        stickpix_bw = (rowsInImage < max(pt[1,1],pt[1,0])) * (rowsInImage > min(pt[1,1],pt[1,0])) * (columnsInImage < pt[0,1] + th) * (columnsInImage > pt[0,1] - th)

    # the brightness of the points on the stick is a function of its

    # distance to the segment

    #[ys,xs] = ind2sub(stickpix_bw)
    idx_bw = np.argwhere(stickpix_bw >0)
    ys = idx_bw[:, 0]
    xs = idx_bw[:, 1]

    px = pt[0,1] - pt[0,0]

    py = pt[1,1] - pt[1,0]

    pp = px*px + py*py

    # the distance between a pixel and the fish backbone

    ysLength = max(ys.shape)

    d_radial = np.zeros((max(ys.shape),1))

    # the distance between a pixel and the anterior end of the

    # segment (0 < d_axial < 1)

    b_axial = np.zeros((max(ys.shape),1))


    for i in range(0,max(ys.shape)):

        u = (((xs[i]+1) - pt[0,0]) * px + ((ys[i]+1) - pt[1,0]) * py) / pp

        dx = pt[0,0] + u * px - xs[i]-1

        dy = pt[1,0] + u * py - ys[i]-1

        d_radial[i] = dx*dx + dy*dy

        b_axial[i] = 1 - (1 - bt_gradient) * u * 0.9

    # b_stick = scipy.stats.norm.pdf(d_radial, 0, th)/p_max * 255
    b_stick = normpdf(d_radial, 0, th)/p_max * 255

    b_stick = uint8(b_stick)

    for i in range(0,max(ys.shape)):
        stickpix[ys[i],xs[i]] = uint8(b_stick[i]*b_axial[i])

    stickpix = stickpix * bt
    stickpix = uint8(stickpix)

    graymodel = np.maximum(ballpix,stickpix)
    return uint8(graymodel)

def gen_lut_s_tail(n,seglenidx,d1,d2,a):



    size_lut = 15

    size_half = (size_lut+1)/2

    imblank = np.zeros((size_lut,size_lut))



    imageSizeX = size_lut

    imageSizeY = size_lut

    random_number = np.random.normal(1.1,0.1)

    random_number = 1.1

    # size of the balls in the model

    temp = [2.5,2.4,2.3,2.2,1.8,1.5,1.3,1.2]
    temp = np.array(temp)
    ballsize = random_number*temp

    # thickness of the sticks in the model
    temp = [8,7,6,5,4,3,2.5,2.5]
    temp = np.array(temp)
    thickness = random_number*temp

    # brightness of the tail

    b_tail = [0.5,0.45,0.4,0.32,0.28,0.24,0.22,0.20]
    b_tail = np.array(b_tail)
    b_tail = random_number * b_tail

    x = np.linspace(1, imageSizeX, imageSizeX)
    y = np.linspace(1, imageSizeY, imageSizeY)

    [columnsInImage0, rowsInImage0] = np.meshgrid(x, y)



    radius = ballsize[n]

    th = thickness[n]

    # p_max = scipy.stats.norm.pdf(0,loc= 0,scale= th)
    p_max = normpdf(0,0,th)

    bt_gradient = b_tail[n]/b_tail[n-1]

    seglen = 0.2 *seglenidx

    bt = b_tail[n-1]*(1 - 0.02*seglenidx)



    centerX = size_half + d1/5

    centerY = size_half + d2/5

    columnsInImage = columnsInImage0

    rowsInImage = rowsInImage0

    ballpix = (rowsInImage - centerY)**2 + (columnsInImage - centerX)**2 <= radius**2

    ballpix = uint8(uint8(uint8(uint8(ballpix) * 255) * bt) * 0.85)

    t = 2*np.pi*(a-1)/180

    pt = np.zeros((2,2))

    R = [[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]

    vec = np.matmul(R,np.array([[seglen] ,[0]]))

    pt[:,0] = np.array([size_half + d1/5, size_half + d2/5])

    pt[:, 1] = pt[:, 0] + vec[:, 0]
    stickpix = imblank

    columnsInImage = columnsInImage0

    rowsInImage = rowsInImage0

    if (pt[0,1] - pt[0,0]) != 0:

        slope = (pt[1,1] - pt[1,0])/(pt[0,1] - pt[0,0])

        # vectors perpendicular to the line segment

        # th is the thickness of the sticks in the model

        vp = np.array([[-slope],[1]]) / np.linalg.norm(np.array([[-slope],[1]]))

        # one vertex of the rectangle

        V1 = pt[:,1] - vp[:,0] * th

        #two sides of the rectangle

        s1 = 2 * vp * th

        s2 = pt[:,0] - pt[:,1]

        # find the pixels inside the rectangle

        r1 = rowsInImage - V1[1]

        c1 = columnsInImage - V1[0]

        # inner products

        ip1 = r1 * s1[1] + c1 * s1[0]

        ip2 = r1 * s2[1] + c1 * s2[0]

        stickpix_bw = (ip1 > 0) * (ip1 < np.dot(s1[:,0], s1[:,0])) * (ip2 > 0) * (ip2 < np.dot(s2,s2))

    else:
        stickpix_bw = (rowsInImage < max(pt[1,1],pt[1,0])) * (rowsInImage > min(pt[1,1],pt[1,0])) * (columnsInImage < pt[0,1] + th) * (columnsInImage > pt[0,1] - th)




    # the brightness of the points on the stick is a function of its

    # distance to the segment

    [ys,xs] = ind2sub(stickpix_bw)

    px = pt[0,1] - pt[0,0]

    py = pt[1,1] - pt[1,0]

    pp = px*px + py*py

    # the distance between a pixel and the fish backbone

    d_radial = np.zeros((max(ys.shape),1))

    # the distance between a pixel and the anterior end of the

    # segment (0 < d_axial < 1)

    b_axial = np.zeros((max(ys.shape),1))

    for i in range(0, max(ys.shape)):

        u = (((xs[i]+1) - pt[0,0]) * px + ((ys[i]+1) - pt[1,0]) * py) / pp

        dx = pt[0,0] + u * px - xs[i]-1

        dy = pt[1,0] + u * py - ys[i]-1

        d_radial[i] = dx*dx + dy*dy

        b_axial[i] = 1 - (1 - bt_gradient) * u * 0.9


    # b_stick = 255 - im2uint8(d_radial/max(d_radial));

    b_stick = normpdf(d_radial, 0, th)/p_max * 255
    b_stick = uint8(b_stick)

    for i in range(0,max(ys.shape)):
        stickpix[ys[i],xs[i]] = uint8(b_stick[i]*b_axial[i])





    stickpix = stickpix * bt
    stickpix = uint8(stickpix)

    graymodel = np.maximum(ballpix,stickpix)
    graymodel = uint8(graymodel)

    return graymodel

def headmodel(x, y, z, seglen, theta1,phi1, brightness, size_lut):
    head_w = seglen * 0.6962
    head_l = seglen * 0.7675 # 0.8196; % 1 / sqrt(2) 7475
    head_l = seglen * 0.8575
    head_h = seglen * 0.6426 # 0.7622; % 0.7 / sqrt(2)
    head_h = seglen * 0.8226
    head_h = seglen * 0.7626
    c_head = 1.1971 # 1.1296;

    pt_original = np.empty((3,3))
    pt_original[:,1] = [size_lut/2, size_lut/2 , size_lut/2]
    pt_original[:,0] = pt_original[:,1] - [seglen, 0, 0]
    temp = Rz(theta1) * Ry(-phi1) * np.array([[seglen], [0], [0]])
    pt_original[:, 2] = pt_original[:, 1] + [temp[0,0],temp[1,0],temp[2,0]]
    # R = rotz(heading) * roty(inclination) * rotx(roll);
    head_c = [c_head * pt_original[0,0] + (1 - c_head) * pt_original[0,1],
    c_head * pt_original[1,0] +
              (1 - c_head) * pt_original[1,1],
    pt_original[2,1] - seglen / (8.3590 + (np.random.rand() - 0.5) * 0.05)] # 3.4609];
    # head_c = head_c - pt_original(:, 2);
    # head_c = R * head_c + pt_original(:, 2);

    XX = x - head_c[0]
    YY = y - head_c[1]
    ZZ = z - head_c[2]

    head_model = np.exp(-2 * (XX * XX / (2 * head_l ** 2) + YY * YY / (2 * head_w ** 2) +
                              ZZ * ZZ / (2 * head_h ** 2) - 1))
    head_model = head_model * brightness

    return head_model

def project_camera_copy(model, X, Y, Z, proj_params, indices, cb, cs1, cs2):

    (coor_b, coor_s1, coor_s2) = calc_proj_w_refra_cpu(np.array([X, Y, Z]), proj_params)

    coor_b[0,:] = coor_b[0,:] - cb[2]
    coor_b[1,:] = coor_b[1,:] - cb[0]
    coor_s1[0,:] = coor_s1[0,:] - cs1[2]
    coor_s1[1,:] = coor_s1[1,:] - cs1[0]
    coor_s2[0,:] = coor_s2[0,:] - cs2[2]
    coor_s2[1,:] = coor_s2[1,:] - cs2[0]

    projection_b = np.zeros((int(cb[1] - cb[0] + 1) , int(cb[3] - cb[2] + 1)))
    projection_s1 = np.zeros((int(cs1[1] - cs1[0] +1),int( cs1[3] - cs1[2] +1)))
    projection_s2 = np.zeros((int(cs2[1] - cs2[0] +1),int( cs2[3] - cs2[2] +1)))

    sz_b = np.shape(projection_b)
    sz_s1 = np.shape(projection_s1)
    sz_s2 = np.shape(projection_s2)

    count_mat_b = np.zeros(np.shape(projection_b)) + 0.0001
    count_mat_s1 = np.zeros(np.shape(projection_s1)) + 0.0001
    count_mat_s2 = np.zeros(np.shape(projection_s2)) + 0.0001



    length = max(indices.shape)

    x = np.linspace(0,length-1,length)
    x = x.astype(int)


    fval = np.logical_or(np.floor(coor_b[1,x]) > sz_b[0]-1, np.floor(coor_b[0,x]) > sz_b[1]-1)
    sval = np.logical_or(np.floor(coor_b[1,x]) < 0,np.floor(coor_b[0,x] ) < 0)

    finval = np.logical_not(np.logical_or(fval,sval))
    model = np.array(model)

    index1 = (np.floor(coor_b[1, x[finval]])).astype(int)
    index2 = (np.floor(coor_b[0, x[finval]])).astype(int)

    values = model[(indices[x[finval]]).astype(int)]

    np.add.at(projection_b, (index1, index2), values)
    np.add.at(count_mat_b, (index1, index2), 1)

    #projection_b = projection_b / count_mat_b
    projection_b = np.divide(projection_b,count_mat_b)





    i = np.linspace(0,length-1,length)
    i = i.astype(int)

    fval = np.logical_or(np.floor(coor_s1[1, i]) > sz_s1[0]-1,np.floor(coor_s1[0, i]) > sz_s1[1]-1)
    sval = np.logical_or(np.floor(coor_s1[1, i]) < 0,np.floor(coor_s1[0, i]) < 0)
    finval = np.logical_not(np.logical_or(fval,sval))

    index1 = (np.floor(coor_s1[1, i[finval]])).astype(int)
    index2 = (np.floor(coor_s1[0, i[finval]])).astype(int)

    values = model[(indices[i[finval]]).astype(int)]

    np.add.at(projection_s1,(index1,index2),values)
    np.add.at(count_mat_s1,(index1,index2),1)

    #projection_s1 = projection_s1 / count_mat_s1
    projection_s1 = np.divide(projection_s1,count_mat_s1)




    x = np.linspace(0, length - 1, length)
    x = x.astype(int)

    fval = np.logical_or(np.floor(coor_s2[1, x]) > sz_s2[0] - 1, np.floor(coor_s2[0, x]) > sz_s2[1] - 1)
    sval = np.logical_or(np.floor(coor_s2[1, x]) < 0, np.floor(coor_s2[0, x]) < 0)

    finval = np.logical_not(np.logical_or(fval, sval))

    index1 = (np.floor(coor_s2[1, x[finval]])).astype(int)
    index2 = (np.floor(coor_s2[0, x[finval]])).astype(int)

    values = model[(indices[x[finval]]).astype(int)]

    np.add.at(projection_s2, (index1, index2), values)
    np.add.at(count_mat_s2, (index1, index2), 1)

    #projection_s2 = projection_s2 / count_mat_s2
    projection_s2 = np.divide(projection_s2,count_mat_s2)

    return projection_b,projection_s1,projection_s2

def reorient_belly_model(model, x_c, y_c, z_c, heading, theta2, inclination, phi2, roll, ref_vec, hinge1, hinge2):
    if (np.any(model) == False):
        length = max(np.shape(x_c))
        indices = np.linspace(0,length-1,num=(length),dtype= int)
    else:
        indices = np.argwhere(model)
        indices = indices[:,0]

    R_canvas = Rz(theta2/2) @ Ry(-phi2/2)

    new_coor = R_canvas * np.array([x_c[indices] - hinge1[0], y_c[indices] - hinge1[1], z_c[indices] - hinge1[2]])

    X = new_coor[0,:] + hinge1[0]
    Y = new_coor[1,:] + hinge1[1]
    Z = new_coor[2,:] + hinge1[2]

    R = Rz(heading) @ Ry(inclination) @ Rx(roll)

    new_coor = R * np.array([X - hinge2[0],
    Y - hinge2[1], Z - hinge2[2]])

    X = new_coor[0,:] + hinge2[0] + ref_vec[0]
    Y = new_coor[1,:] + hinge2[1] + ref_vec[1]
    Z = new_coor[2,:] + hinge2[2] + ref_vec[2]

    return X, Y, Z, indices

def reorient_model(model, x_c, y_c, z_c, heading, inclination, roll, ref_vec, hinge):

    if (np.any(model) == False):
        length = max(np.shape(x_c))
        indices = np.linspace(0,length-1,num=(length),dtype= int)
    else:
        indices = np.argwhere(model)
        indices = indices[:,0]

    R = Rz(heading) @ Ry(inclination) @ Rx(roll)

    new_coor = R @ np.array([x_c[indices] - hinge[0], y_c[indices] - hinge[1], z_c[indices] - hinge[2]])

    X = new_coor[0,:] + hinge[0] + ref_vec[0]
    Y = new_coor[1,:] + hinge[1] + ref_vec[1]
    Z = new_coor[2,:] + hinge[2] + ref_vec[2]

    return X, Y, Z, indices

def view_b_lut_new_real_cpu(crop_coor, pt, projection, imageSizeX, imageSizeY):


    vec_pt = pt[:,1: 10] - pt[:, 0: 9]

    segslen = (np.sum(vec_pt * vec_pt, 0))**(1/2)

    segslen = np.tile(segslen, (2, 1))
    vec_pt_unit = vec_pt / segslen

    theta_prime = np.arctan2(vec_pt_unit[1,:],vec_pt_unit[0,:])
    theta = np.zeros((2,max(theta_prime.shape)))
    theta[0,:] = theta_prime
    theta[1,:] = theta_prime

    #shift pts t0 the cropped images

    pt[0,:] = pt[0,:] - crop_coor[2] + 1

    pt[1,:] = pt[1,:] - crop_coor[0] + 1

    imageSizeY = crop_coor[1] - crop_coor[0] + 1
    imageSizeX = crop_coor[3] - crop_coor[2] + 1

    imblank = np.zeros((int(imageSizeY), int(imageSizeX)))

    bodypix = imblank


    headpix = uint8(uint8(projection / 2) * 5.2)

    size_lut = 19

    size_half = (size_lut + 1) / 2


    coor_t = np.floor(pt)

    dt = np.floor((pt - coor_t) * 5) + 1

    at = np.mod(np.floor(theta * 180 / np.pi), 360) + 1

    seglen = segslen

    indices = np.argwhere(seglen<3.3)
    for index in indices:
        seglen[index[0],index[1]] = 3.2

    indices = np.argwhere(seglen>10.5)
    for index in indices:
        seglen[index[0],index[1]] =10.6


    seglenidx = roundHalfUp((seglen - 5) / 0.2)

    #Uncomment For the fixed version
    seglenidx = seglenidx[0,:]
    for ni in range(0,7):

        n = ni + 2

        tailpix = imblank

        #tail_model = gen_lut_b_tail.gen_lut_b_tail(ni + 1, seglenidx[n], dt[0, n], dt[1, n], at[0,n])
        #newIndex = linIndxTo2DIndx(n+1,seglenidx.shape)
        #tail_model = gen_lut_b_tail(ni + 1, seglenidx[newIndex], dt[0, n], dt[1, n], at[0,n])
        tail_model = gen_lut_b_tail(ni + 1, seglenidx[n], dt[0, n], dt[1, n], at[0,n])

        try:
            tailpix[int(max(1, coor_t[1, n] - (size_half - 1))) - 1: int(min(imageSizeY, coor_t[1, n] + (size_half - 1))), int(max(1, coor_t[0, n] - (size_half - 1))) - 1: int(min(imageSizeX, coor_t[0, n] + (size_half - 1)))] = tail_model[int(max((size_half + 1) - coor_t[1, n], 1))-1: int(min(imageSizeY - coor_t[1, n] + size_half, size_lut)), int(max((size_half + 1 ) - coor_t[0, n], 1))-1: int(min(imageSizeX - coor_t[0, n] + size_half, size_lut))]
        except:
            continue
        bodypix = np.maximum(bodypix, tailpix)
        bodypix = uint8(bodypix)

    graymodel = np.maximum(headpix, uint8((.7) * bodypix))
    graymodel = uint8(graymodel)

    return graymodel



def view_s_lut_new_real_cpu(crop_coor, pt,projection,imageSizeX,imageSizeY):

    # Find the coefficients of the line that defines the refracted ray
    vec_pt = pt[:,1:10] - pt[:,0:9]

    segslen = (np.sum(vec_pt*vec_pt,0))**(1/2)

    segslen = np.tile(segslen, (2, 1))

    vec_pt_unit = vec_pt /segslen

    theta_prime = np.arctan2(vec_pt_unit[1, :], vec_pt_unit[0, :])
    theta = np.zeros((2, max(theta_prime.shape)))
    theta[0, :] = theta_prime
    theta[1:] = theta_prime



    # shift pts to the cropped images

    pt[0,:] = pt[0,:] - crop_coor[2] + 1

    pt[1,:] = pt[1,:] - crop_coor[0] + 1

    imageSizeY = crop_coor[1] - crop_coor[0] + 1

    imageSizeX = crop_coor[3] - crop_coor[2] + 1

    ###These three are all uint8
    imblank = np.zeros((int(imageSizeY),int(imageSizeX)))

    imblank_cpu = np.zeros((int(imageSizeY),int(imageSizeX)))

    bodypix = imblank_cpu

    headpix = uint8(uint8(projection/1.8)*5.2)

    # tail

    size_lut = 15

    size_half = (size_lut+1)/2

    seglen = segslen


    seglen[seglen<0.2] = 0.1



    seglen[seglen>7.9] = 8



    seglenidx = roundHalfUp(seglen/0.2)

    coor_t = np.floor(pt)

    dt = np.floor((pt - coor_t)*5) + 1

    at = np.mod(np.floor(theta*90/np.pi),180) + 1

    seglenidx = seglenidx[0,:]
    for ni in range(0,7):

        n = ni+2

        tailpix = imblank

        #newIndex = linIndxTo2DIndx(n+1, seglenidx.shape)

        tail_model = gen_lut_s_tail(ni + 1, seglenidx[n], dt[0,n], dt[1,n], at[0,n])
        #tail_model = gen_lut_s_tail(ni + 1, seglenidx[newIndex], dt[0, n], dt[1, n], at[0,n])
        
        try:
            tailpix[int(max(1, coor_t[1, n] - (size_half - 1))) - 1: int(min(imageSizeY, coor_t[1, n] + (size_half - 1))), int(max(1, coor_t[0, n] - (size_half - 1))) - 1: int(min(imageSizeX, coor_t[0, n] + (size_half - 1)))] = tail_model[int(max((size_half + 1) - coor_t[1, n], 1))-1: int(min(imageSizeY - coor_t[1, n] + size_half, size_lut)), int(max((size_half + 1 ) - coor_t[0, n], 1))-1: int(min(imageSizeX - coor_t[0, n] + size_half, size_lut))]
        except:
            continue
        bodypix = np.maximum(bodypix, tailpix)




    graymodel = np.maximum(headpix,(0.8)*bodypix)
    graymodel = uint8(graymodel)

    return graymodel

def return_head_real_model_new(x,fishlen,proj_params,cb,cs1,cs2):


    # Calculate the 3D points pt from model parameters
    seglen = fishlen * 0.09
    size_lut_3d = 2 # Represents the length of the box in which the 3D fish is constructed
    inclination = x[12]
    heading = x[3]
    hp = np.array([[x[0]],[x[1]],[x[2]]])
    dtheta = x[3:12]
    theta = np.cumsum(dtheta)
    dphi = x[12:21]
    phi = np.cumsum(dphi)
    roll = x[21]

    vec_unit = seglen* np.array([[np.cos(theta) * np.cos(phi)], [np.sin(theta) * np.cos(phi)], [-np.sin(phi)]])
    vec_unit = vec_unit[:,0,:]
    # vec_ref_1 is parallel to the camera sensor of b and s2
    # vec_ref_2 is parallel to s1
    vec_ref_1 = np.array([[seglen], [0], [0]])
    vec_ref_2 = np.array([[0], [seglen], [0]])
    pt_ref = np.array([hp + vec_ref_1, hp + vec_ref_2])
    pt_ref = np.transpose(pt_ref[:, :, 0])

    z = np.array([[0], [0], [0]])
    temp = np.concatenate((z, vec_unit), axis=1)
    pt = np.cumsum(temp, axis=1)

    pt = np.concatenate((pt + np.tile(hp, (1, 10)), pt_ref), axis=1)

    # Construct the larva
    # Locate center of the head to use as origin for rotation of the larva.
    # This is consistent with the way in which the parameters of the model are
    # computed during optimization
    resolution = 75
    x_c = (np.linspace(0, 1, num=resolution) * size_lut_3d)
    y_c = (np.linspace(0, 1, num=resolution) * size_lut_3d)
    z_c = (np.linspace(0, 1, num=resolution) * size_lut_3d)

    [x_c, y_c, z_c] = np.meshgrid(x_c, y_c, z_c)

    x_c = x_c.transpose()
    x_c = x_c.flatten()

    y_c = y_c.transpose()
    y_c = y_c.flatten()

    z_c = z_c.transpose()
    z_c = z_c.flatten()

    pt_original = np.zeros((3,3))
    pt_original[:,1] = np.array([size_lut_3d/2, size_lut_3d/2, size_lut_3d/2])
    pt_original[:,0] = pt_original[:,1] - np.array([seglen,0,0])

    #This line was modified with the use of temp
    temp = Rz(x[4]) * Ry(x[13]) * np.array([[seglen], [0], [0]])
    pt_original[:,2] = pt_original[:,1] + np.array([temp[0,0],temp[1,0],temp[2,0]])
    hinge = pt_original[:,2] # COM of the fish
    # Calculate the 3D fish
    # eye_br = 120.8125; % 150.8125;
    # head_br = 15.953318957123471;
    # belly_br = 17.05897936938326;
    # eye_br = eye_br*4.1; head_br = head_br*3.1; belly_br = belly_br*3.6;
    # %eye_br = eye_br*0.033; head_br = head_br*0.021; belly_br = belly_br*0.028;
    #%% Translate the model to overlap with the cropped image
    vec_13 = hp - pt[:,2].reshape(-1,1)
    vec_13 = np.tile(vec_13,(1,12))
    pt = pt + vec_13
    ref_vec = pt[:,2] - hinge
    eye_br = 13
    head_br = 13
    belly_br = 13
    # Render and project the 3D fish

    random_vector_eye = np.random.rand(5)

    [eye1_model,eye1_c] = eye1model(x_c, y_c, z_c, seglen, x[4], x[13], eye_br, size_lut_3d, random_vector_eye)
    [eye2_model,eye2_c] = eye2model(x_c, y_c, z_c, seglen, x[4], x[13], eye_br, size_lut_3d, random_vector_eye)
    [belly_model, belly_c] = bellymodel(x_c, y_c, z_c, seglen, x[4], x[13], belly_br, size_lut_3d)
    head_model = headmodel(x_c, y_c, z_c, seglen, x[4], x[13], head_br, size_lut_3d)

    # Eye1
    [model_X, model_Y, model_Z, indices] = reorient_model(eye1_model,x_c,y_c,z_c,heading,inclination,roll,ref_vec,hinge)

    #This is necessay for now, can be made better later
    model_X = np.array(model_X)
    model_X = model_X[0, :]

    model_Y = np.array(model_Y)
    model_Y = model_Y[0, :]
    model_Z = np.array(model_Z)
    model_Z = model_Z[0, :]

    [eye1_b, eye1_s1, eye1_s2] = project_camera_copy(eye1_model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2)


    # Eye2
    [model_X, model_Y, model_Z, indices] = reorient_model(eye2_model,x_c,y_c,z_c,heading,inclination,roll,ref_vec,hinge)
    model_X = np.array(model_X)
    model_X = model_X[0, :]
    model_Y = np.array(model_Y)
    model_Y = model_Y[0, :]
    model_Z = np.array(model_Z)
    model_Z = model_Z[0, :]
    [eye2_b, eye2_s1, eye2_s2] = project_camera_copy(eye2_model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2)

    # Head
    [model_X, model_Y, model_Z, indices] = reorient_model(head_model,x_c,y_c,z_c,heading,inclination,roll,ref_vec,hinge)
    model_X = np.array(model_X)
    model_X = model_X[0, :]
    model_Y = np.array(model_Y)
    model_Y = model_Y[0, :]
    model_Z = np.array(model_Z)
    model_Z = model_Z[0, :]
    [head_b, head_s1, head_s2] = project_camera_copy(head_model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2)

    # Bellymodel
    [model_X, model_Y, model_Z, indices] = reorient_belly_model(belly_model,
                                                                x_c,y_c,z_c,heading,x[4]+0.3*x[5],inclination,x[13]+0.3*x[14],roll,ref_vec,pt_original[:,0],hinge)
    model_X = np.array(model_X)
    model_X = model_X[0, :]
    model_Y = np.array(model_Y)
    model_Y = model_Y[0, :]
    model_Z = np.array(model_Z)
    model_Z = model_Z[0, :]

    [belly_b, belly_s1, belly_s2] = project_camera_copy(belly_model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2);


    eye_br = 112
    head_br = 67
    belly_br = 85


    eye1_b = mat2gray(eye1_b)
    eye2_b = mat2gray(eye2_b)
    eye1_b = 2*(sigmoid(eye1_b, 2.5) - 0.5)*eye_br
    eye2_b = 2*(sigmoid(eye2_b, 2.5) - 0.5)*eye_br
    #head_b = head_b*(head_br/double(max(max(head_b))))
    head_b = mat2gray(head_b)
    head_b = 2*(sigmoid(head_b,3) - 0.5)*head_br
    #belly_b = belly_b*(belly_br/double(max(max(belly_b))));
    belly_b = mat2gray(belly_b)
    belly_b = 2*(sigmoid(belly_b,3) - 0.5)*belly_br

    #eye_scaling = max(eye_br/double(max(max(eye1_s1))),eye_br/double(max(max(eye2_s1))));
    #eye1_s1 = eye1_s1*(eye_br/double(max(max(eye1_s1))));

    #pdb.set_trace()
    eye1_s1 = mat2gray(eye1_s1)
    
    eye2_s1 = mat2gray(eye2_s1)
    eye1_s1 = 2*(sigmoid(eye1_s1,2.8) - 0.5)*eye_br
    eye2_s1 = 2*(sigmoid(eye2_s1,2.8) - 0.5)*eye_br
    #eye2_s1 = eye2_s1*(eye_br/double(max(max(eye2_s1))));
    #head_s1 = head_s1*(head_br/double(max(max(head_s1))));
    head_s1 = mat2gray(head_s1)
    head_s1 = 2*(sigmoid(head_s1,2) - 0.5)*head_br
    #belly_s1 = belly_s1*(belly_br/double(max(max(belly_s1))));
    belly_s1 = mat2gray(belly_s1)
    belly_s1 = 2*(sigmoid(belly_s1,2) - 0.5)*belly_br

    # eye_scaling = max(eye_br/double(max(max(eye1_s2))),eye_br/double(max(max(eye2_s2))));
    #eye1_s2 = eye1_s2*(eye_br/double(max(max(eye1_s2))));
    #eye2_s2 = eye2_s2*(eye_br/double(max(max(eye2_s2))));
    eye1_s2 = mat2gray(eye1_s2)
    eye2_s2 = mat2gray(eye2_s2)
    eye1_s2 = 2*(sigmoid(eye1_s2,2.8) - 0.5)*eye_br
    eye2_s2 = 2*(sigmoid(eye2_s2,2.8) - 0.5)*eye_br
    #head_s2 = head_s2*(head_br/double(max(max(head_s2))));
    head_s2 = mat2gray(head_s2)
    head_s2 = 2*(sigmoid(head_s2,2) - 0.5)*head_br
    #belly_s2 = belly_s2*(belly_br/double(max(max(belly_s2))));
    belly_s2 = mat2gray(belly_s2)
    belly_s2 = 2*(sigmoid(belly_s2,2) - 0.5)*belly_br



    graymodel_b = np.maximum(np.maximum(np.maximum(eye1_b,eye2_b),head_b),belly_b)
    graymodel_s1 = np.maximum(np.maximum(np.maximum(eye1_s1,eye2_s1),head_s1),belly_s1)
    graymodel_s2 = np.maximum(np.maximum(np.maximum(eye1_s2,eye2_s2),head_s2),belly_s2)

    [eyeCenters_X, eyeCenters_Y, eyeCenters_Z, throwaway] = reorient_model(np.array([]),
                np.array([eye1_c[0], eye2_c[0]]), np.array([eye1_c[1], eye2_c[1]]),
                np.array([eye1_c[2], eye2_c[2]]), heading, inclination, roll, ref_vec, hinge)

    eyeCenters_X = np.array(eyeCenters_X)
    eyeCenters_X = eyeCenters_X[0, :]

    eyeCenters_Y = np.array(eyeCenters_Y)
    eyeCenters_Y = eyeCenters_Y[0, :]

    eyeCenters_Z = np.array(eyeCenters_Z)
    eyeCenters_Z = eyeCenters_Z[0, :]


    [eye_b, eye_s1, eye_s2] = calc_proj_w_refra_cpu(np.array([eyeCenters_X, eyeCenters_Y, eyeCenters_Z]), proj_params)
    eye_3d_coor = np.array([eyeCenters_X, eyeCenters_Y, eyeCenters_Z])

    return graymodel_b, graymodel_s1, graymodel_s2, eye_b, eye_s1, eye_s2, eye_3d_coor


def return_graymodels_fish(x,crop_b,crop_s1,crop_s2,proj_params,fishlen,imageSizeX,imageSizeY):
    seglen = fishlen*0.09
    hp = np.array([[x[0]], [x[1]], [x[2]]])
    dtheta = x[4:12]

    theta = np.cumsum(np.concatenate(([0],dtheta)))
    dphi = x[13:21]
    phi = np.cumsum(np.concatenate(([0],dphi)))

    vec = seglen*np.array([np.cos(theta)*np.cos(phi), np.sin(theta) *np.cos(phi), -np.sin(phi)])

    theta_0 = x[3]
    phi_0 = x[12]
    gamma_0 = x[21]

    vec_ref_1 = np.array([[seglen], [0], [0]])
    vec_ref_2 = np.array([[0], [seglen], [0]])

    pt_ref = np.concatenate((hp + vec_ref_1, hp + vec_ref_2), axis=1)

    z = np.array([[0], [0], [0]])
    frank = np.concatenate((z, vec), axis=1)
    pt = np.cumsum(frank,axis=1)

    pt = Rz(theta_0) @ Ry(phi_0) @ Rx(gamma_0) @ pt

    pt = np.array(pt)
    pt = np.concatenate((pt + np.tile(hp, (1, 10)),pt_ref),axis=1)

    hinge = pt[:, 2]
    vec_13 = pt[:, 0] - hinge

    vec_13 = np.array([[vec_13[0]],[vec_13[1]],[vec_13[2]]])
    vec_13 = np.tile(vec_13, (1, 12))

    pt = pt + vec_13
    pt = np.array(pt)
    [coor_b,coor_s1,coor_s2] = calc_proj_w_refra_cpu(pt, proj_params)

    #There is a better way to do this I will comeback to it later
    coor_b_shape = np.shape(coor_b)
    coor_b = coor_b[:, 0:coor_b_shape[1] - 2]

    idxs = [*range(coor_s1.shape[1])]
    idxs.pop(coor_s1.shape[1] - 2)  # this removes elements from the list
    coor_s1 = coor_s1[:, idxs]

    coor_s2 = coor_s2[:, 0:coor_s2.shape[1]-1]

    [projection_b, projection_s1, projection_s2, eye_b, eye_s1, eye_s2, eye_coor_3d] = \
    return_head_real_model_new(x, fishlen, proj_params, crop_b, crop_s1, crop_s2)

    ######Should Pass in copys of arrays#############
    temp = np.copy(coor_b)
    gray_b = view_b_lut_new_real_cpu(crop_b, temp, projection_b, imageSizeX, imageSizeY)
    temp = np.copy(coor_s1)
    gray_s1 = view_s_lut_new_real_cpu(crop_s1, temp, projection_s1, imageSizeX, imageSizeY)
    temp = np.copy(coor_s2)
    gray_s2 = view_s_lut_new_real_cpu(crop_s2, temp, projection_s2, imageSizeX, imageSizeY)

    annotated_b = np.zeros((2, coor_b.shape[1]))
    annotated_b[0, :] = coor_b[0, :] - crop_b[2] 
    annotated_b[1, :] = coor_b[1, :] - crop_b[0] 

    annotated_s1 = np.zeros((2, coor_s1.shape[1]))
    annotated_s1[0, :] = coor_s1[0, :] - crop_s1[2] 
    annotated_s1[1, :] = coor_s1[1, :] - crop_s1[0] 

    annotated_s2 = np.zeros((2, coor_s2.shape[1]))
    annotated_s2[0, :] = coor_s2[0, :] - crop_s2[2] 
    annotated_s2[1, :] = coor_s2[1, :] - crop_s2[0] 

    annotated_b = annotated_b[:, 0:10]
    annotated_s1 = annotated_s1[:, 0:10]
    annotated_s2 = annotated_s2[:, 0:10]

    eye_b[0, :] = eye_b[0, :] - crop_b[2] 
    eye_b[1, :] = eye_b[1, :] - crop_b[0] 
    eye_s1[0, :] = eye_s1[0, :] - crop_s1[2] 
    eye_s1[1, :] = eye_s1[1, :] - crop_s1[0] 
    eye_s2[0, :] = eye_s2[0, :] - crop_s2[2] 
    eye_s2[1, :] = eye_s2[1, :] - crop_s2[0] 

    eye_b = eye_b 
    eye_s1 = eye_s1 
    eye_s2 = eye_s2 

    annotated_b = annotated_b 
    annotated_s1 = annotated_s1
    annotated_s2 = annotated_s2 

    crop_b = crop_b 
    crop_s1 = crop_s11
    crop_s2 = crop_s2 

    coor_3d = pt[:, 0:10]

    coor_3d = np.concatenate((coor_3d, eye_coor_3d), axis=1)

    return gray_b, gray_s1, gray_s2,annotated_b, annotated_s1, annotated_s2, eye_b, eye_s1, eye_s2,coor_3d
