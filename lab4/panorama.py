import sys
import numpy as np
import cv2
from numba import njit
from time import time 
from trws_algorithm import *


def get_matches(src, dst):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(src,None)
    kp2, des2 = sift.detectAndCompute(dst,None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    return src_pts, dst_pts

def interpolate(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T

def apply_homography(src,H,img_size):
    h,w = img_size
    # inverse homography matrix to apply it to output image
    H = np.linalg.inv(H)
    coords = np.zeros((h,w,2))
    for i in range(h):
        for j in range(w):
            unnorm = H@np.array([j,i,1])
            coords[i,j,:] = (unnorm/unnorm[-1])[:-1]
    # new coordimates after homography
    coords = coords.reshape(-1,2)
    # interpolate colors
    warped = interpolate(src, coords[:,0], coords[:,1])
    warped = warped.reshape(h,w,3)
    warped = warped.astype(np.uint8)
    return warped

def calculate_homography(p1,p2):

    # calculate homography matrix from 4 correspondent points
    A = np.array([
        [p1[0][0][0], p1[0][0][1], 1, 0, 0, 0, -p2[0][0][0]*p1[0][0][0], -p2[0][0][0]*p1[0][0][1], -p2[0][0][0]],
        [0, 0, 0, p1[0][0][0], p1[0][0][1], 1, -p2[0][0][1]*p1[0][0][0], -p2[0][0][1]*p1[0][0][1], -p2[0][0][1]],
        
        [p1[1][0][0], p1[1][0][1], 1, 0, 0, 0, -p2[1][0][0]*p1[1][0][0], -p2[1][0][0]*p1[1][0][1], -p2[1][0][0]],
        [0, 0, 0, p1[1][0][0], p1[1][0][1], 1, -p2[1][0][1]*p1[1][0][0], -p2[1][0][1]*p1[1][0][1], -p2[1][0][1]],
        
        [p1[2][0][0], p1[2][0][1], 1, 0, 0, 0, -p2[2][0][0]*p1[2][0][0], -p2[2][0][0]*p1[2][0][1], -p2[2][0][0]],
        [0, 0, 0, p1[2][0][0], p1[2][0][1], 1, -p2[2][0][1]*p1[2][0][0], -p2[2][0][1]*p1[2][0][1], -p2[2][0][1]],
        
        [p1[3][0][0], p1[3][0][1], 1, 0, 0, 0, -p2[3][0][0]*p1[3][0][0], -p2[3][0][0]*p1[3][0][1], -p2[3][0][0]],
        [0, 0, 0, p1[3][0][0], p1[3][0][1], 1, -p2[3][0][1]*p1[3][0][0], -p2[3][0][1]*p1[3][0][1], -p2[3][0][1]]
    ])
        
    _, _, Vh = np.linalg.svd(A)
    # last element = 1 
    H = (Vh[-1,:] / Vh[-1,-1]).reshape(3,3)
    return H


def ransac(src_pts, dst_pts, ransacReprojThreshold=5, maxIters=2000):

    # implementation of ransac algorithm for finding best homography matrix

    # adding ones array to imitate 3D
    src_pts = np.dstack(
        (src_pts, np.ones((src_pts.shape[0], src_pts.shape[1]))))
    dst_pts = np.dstack(
        (dst_pts, np.ones((dst_pts.shape[0], dst_pts.shape[1]))))
    best_count_matches = 0
    for iteration in range(maxIters):
        # get four random points
        random_indices = np.random.randint(0, len(src_pts)-1, 4)
        random_kp_src, random_kp_dst = src_pts[random_indices], dst_pts[random_indices]
        # calculate a homography
        H_current = calculate_homography(random_kp_src, random_kp_dst)
        count_matches = 0
        # check all points
        for i in range(len(src_pts)):
            unnorm = H_current@src_pts[i][0]
            normalized = (unnorm/unnorm[-1])
            # calculate norm
            normalized = custom_norm(normalized-dst_pts[i][0])
            # check solution
            if normalized < ransacReprojThreshold:
                count_matches += 1
        if count_matches >= best_count_matches:
            # update best homography if better match found
            best_count_matches = count_matches
            H_best = H_current
    return H_best


def perpective_transform(points, H):
    # transforms input points using homography matrix
    new_points = []
    for point in points:
        p = H@np.array([point[0][0],point[0][1],1])
        p = p/p[-1]
        new_points.append(p[:-1])
    new_points = np.array(new_points, dtype = np.float32).reshape(points.shape)
    return new_points

def get_H(img1,img2):
    # calculates homography for 2 images
    src_pts, dst_pts = get_matches(img1, img2)
    H_best = ransac(src_pts, dst_pts, ransacReprojThreshold = 5, maxIters = 3000)
    return H_best

def aligning(input_images):
    number_of_images = len(input_images)
    base_image = input_images[0]

    edge_points = np.array([[0,0],
                           [0,base_image.shape[0]-1],
                           [base_image.shape[1]-1,0],
                           [base_image.shape[0]-1,base_image.shape[1]-1]]
                         ).reshape(-1, 1, 2)
    merged_points = np.zeros((number_of_images,4,1,2))
    merged_points[0,...] = edge_points

    homography = []
    for i,image in enumerate(input_images[1:]):
        H = get_H(image, base_image)
        homography.append(H)
        merged_points[i+1,...] = perpective_transform(edge_points,H)
    # edge points from all images
    merged_points = merged_points.reshape(-1,2)

    min_x, max_x = int(min(merged_points[:,0])), int(max(merged_points[:,0]))
    min_y, max_y = int(min(merged_points[:,1])), int(max(merged_points[:,1]))

    aligned_base = np.zeros((max_y-min_y,max_x-min_x,3),dtype=int)
    aligned_base[-min_y:-min_y +base_image.shape[0],-min_x:-min_x + base_image.shape[1]] = base_image 
    # shift matrix
    H_translation = np.array([[1, 0, -min_x],
                              [0, 1, -min_y],
                              [0, 0, 1]])

    aligned_images = [aligned_base]
    for i,image in enumerate(input_images[1:]):
        aligned_image = apply_homography(image,H_translation@homography[i],(max_y-min_y,max_x-min_x))
        aligned_images.append(aligned_image)
    # list of aligned images
    return aligned_images

@njit(fastmath=True)
def calculate_q(images):
    # 0 - if (i,j) can take from k-image, else -inf
    number_of_images = len(images)
    Q = np.full((images[0].shape[0],images[0].shape[1],number_of_images),-np.inf)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            for k in range(Q.shape[2]):
                if (images[k][i,j] != 0).any():
                    Q[i,j,k] = 0
            if (images[:,i,j,:] == 0).all():
                Q[i,j,:] = 0
    return Q


@njit(fastmath=True)
def custom_norm(vec):
    # L2 - norm for 3D vectors
    return np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

@njit(fastmath=True)
def calculate_g(images):
    
    height,width,_ = images[0].shape
    number_of_images = len(images)
    g = np.zeros((height,width,4,number_of_images,number_of_images))
    
    # for each pixel in image
    for i in range(height):
        for j in range(width):
            nbs, _ , nbs_indices = get_neighbours(height, width,i,j)
            # for each neighbour
            for n,[n_i,n_j] in enumerate(nbs):
                for k in range(number_of_images):
                    for k1 in range(number_of_images):
                        if k != k1:
                            g[i,j,nbs_indices[n],k,k1] =  - custom_norm(images[k,i,j]-images[k1,i,j]) - custom_norm(images[k,n_i,n_j]-images[k1,n_i,n_j]) 
    return g

@njit(fastmath = True)
def mapping(images, labelling):
    # map color for each label
    panorama = np.zeros_like(images[0])
    for i in range(panorama.shape[0]):
        for j in range(panorama.shape[1]):
            panorama[i,j,:] = images[labelling[i,j],i,j,:]
    return panorama.astype(np.int32)

def create_panorama(aligned_images, trws_n_iter):
    # calcuate input parameters
    images = np.array(aligned_images, dtype=np.float32)
    Q = calculate_q(images)
    g = calculate_g(images)
    # stitch 2 images using TRW-S
    labelling = optimal_labelling(Q,g,trws_n_iter)
    panorama = mapping(images, labelling)

    return panorama



def main():

    '''
    Parameters
        argv[1]: str 
            trws_n_iter - number of iterations for TRW-S algorithm
        argv[2:]: list
            list of input images
    '''

    # initialise parameters
    trws_n_iter = sys.argv[1]
    images_paths = sys.argv[2:]
    trws_n_iter = int(trws_n_iter)

    input_images = []
    for path in images_paths:
        input_images.append(cv2.imread(path))


    a = time()
    aligned_images = aligning(input_images)
    panorama = create_panorama(aligned_images, trws_n_iter)
    cv2.imwrite("panorama.png",panorama)
    print("total time: ", time() -a)

if __name__ == "__main__":
    main()
