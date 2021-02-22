import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_matches(src, dst):
    '''
    get matched points from 2 images
    '''
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
    
    # all neighbours coordinates 
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1 

    x0, x1 = np.clip(x0, 0, im.shape[1]-1), np.clip(x1, 0, im.shape[1]-1)
    y0, y1 = np.clip(y0, 0, im.shape[0]-1), np.clip(y1, 0, im.shape[0]-1)

    # weighted sum of 4 nearest neighbours
    weighted_sum = (im[y0,x0].T*(x1-x) * (y1-y)).T + (im[y1,x0].T*(x1-x) * (y-y0)).T + (im[y0,x1].T*(x-x0) * (y1-y)).T + (im[y1,x1].T*(x-x0) * (y-y0)).T
    
    return weighted_sum



def apply_homography(H,src):

    # inverse homography matrix to apply it to output image
    H = np.linalg.inv(H)
    coords = np.zeros((src.shape[0],src.shape[1],2))
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            coords[i,j,:] = (H@np.array([i,j,1]))[:-1] 
    # new coordimates after homography
    coords = coords.reshape(-1,2)
    # interpolate colors
    warped = interpolate(src, coords[:,1], coords[:,0])
    warped = warped.reshape(src.shape)
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

def ransac(src_pts, dst_pts, ransacReprojThreshold = 5, maxIters = 2000):

    # implementation of ransac algorithm for finding best homography matrix

    # adding ones array to imitate 3D
    src_pts = np.dstack((src_pts,np.ones((src_pts.shape[0],src_pts.shape[1])) ) )
    dst_pts = np.dstack((dst_pts,np.ones((dst_pts.shape[0],dst_pts.shape[1])) ) )
    best_count_matches = 0
    for iteration in range(maxIters):
        # get four random points
        random_indices = np.random.randint(0,len(src_pts)-1,4)
        random_kp_src, random_kp_dst  = src_pts[random_indices], dst_pts[random_indices]
        # calculate a homography
        H_current = calculate_homography(random_kp_src,random_kp_dst)
        count_matches = 0
        # check all points
        for i in range(len(src_pts)):
            unnorm = H_current@src_pts[i][0]
            normalized = (unnorm/unnorm[-1])
            # calculate norm
            norm = np.linalg.norm(normalized-dst_pts[i][0])
            # check solution
            if norm < ransacReprojThreshold:
                count_matches += 1
        if count_matches >= best_count_matches:
            # update best homography if better match found
            best_count_matches = count_matches
            H_best = H_current.copy()
    return H_best






def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("left_image", type=str, help="path to left image")
    parser.add_argument("right_image", type=str, help="path to right image")
    parser.add_argument("ransac_max_iter", type=int, help="max iteration for ransac algorithm")
    parser.add_argument("output_image", type=str, help="path to output image")
    args = parser.parse_args()


    img1 = cv2.imread(args.left_image)
    img2 = cv2.imread(args.right_image)
    src_pts, dst_pts = get_matches(img1, img2)


    H_best = ransac(src_pts, dst_pts, ransacReprojThreshold = 3, maxIters = args.ransac_max_iter)


    warped_image = apply_homography(H_best,img1)

    # save output image to file
    cv2.imwrite(args.output_image,warped_image)



if __name__ == "__main__":
    main()
