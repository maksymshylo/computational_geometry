import argparse
import numpy as np
from numpy import trace
import PIL.Image
import PIL.ExifTags
import cv2
from numpy.linalg import det, matrix_rank as rank
from fundamental_matrix import *

def get_exif_info(path_to_img):
    """
    parse metadata information from image : focal length, height, width
    """
    img = PIL.Image.open(path_to_img)
    exif_data = img._getexif()
    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in img._getexif().items()
        if k in PIL.ExifTags.TAGS
        }
    return exif['FocalLength'], exif['ExifImageHeight'], exif['ExifImageWidth']

def calculate_K(focal_length, pixel_size, height, width):
    """
    create camera intrinsics matrix
    """
    f = (focal_length/pixel_size )*(10**3)
    K =  np.array([[f, 0, (width/2) ],
                   [0, f, (height/2)],
                   [0, 0, 1]])
    return K

def constraint(E):
    """
    check essential matrix constraint
    """
    return 2* E @ (E.T)@E - trace(  E @ (E.T) )*E

def find_essential_matrix(F, K):
    """
    find essential matrix from fundamental and 
    camera intrinsics parameters
    """
    # calculate Essential matrix
    E = K.T@F@K
    U, S, Vt = np.linalg.svd(E)
    # update sigma to make sigma1=sigma2>0 and sigma3=0
    S1 = np.identity(E.shape[0])
    S1 = (S[0]+S[1])*0.5*S1
    S1[-1,-1] = 0
    updated_E = U@S1@Vt
    # calculate translation and rotation
    t = ((S1[0,0] + S1[1,1])*0.5)*det(U)*Vt[:,2]
    W = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    R = det(U)*U@W@Vt*det(Vt)
    
    return updated_E, rank(updated_E), S, np.diagonal(S1), constraint(updated_E), t, R


def main():

    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("left_image",  type=str,   help="left input image")
    parser.add_argument("right_image", type=str,   help="right input image")
    parser.add_argument("epsilon",     type=int,   help="threshold parameter in PIXELS")
    parser.add_argument("n_iter",      type=int,   help="number of iterations for RANSAC \
                                                         fundamental matrix finding")
    parser.add_argument("pixel_size",  type=float, help="size of pixel in microns")


    args = parser.parse_args()

    img_left  = cv2.imread(args.left_image,  cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(args.right_image, cv2.IMREAD_GRAYSCALE)


    src_pts, dst_pts = get_matches(img_left, img_right)

    src_pts_vis = src_pts.copy()
    dst_pts_vis = dst_pts.copy()
    # add ones to make all coordinates homogeneous
    ones_arr = np.ones((src_pts.shape[0],1))

    src_pts = np.dstack((src_pts, ones_arr))
    dst_pts = np.dstack((dst_pts, ones_arr))

    F, _ = calculate_fundamental(src_pts, dst_pts, args.epsilon, args.n_iter)
    mask = get_mask(src_pts, dst_pts, F, args.epsilon)

    epilines1, epilines2 = get_epilines_images(src_pts_vis, dst_pts_vis, img_left, img_right, mask, F)

    cv2.imwrite("epilines1.png", epilines1)
    cv2.imwrite("epilines2.png", epilines2)


    focal_length, height, width = get_exif_info(args.left_image)
    K = calculate_K(focal_length, args.pixel_size, height, width)

    updated_E, rank_E, sigma, updated_sigma, constraint_updated_E, t, R =\
        find_essential_matrix(F, K)

    print("Essential matrix:  \n", updated_E)
    print("\n\n")
    print("Rank: \n", rank_E)
    print("\n\n")
    print("Original sigma values: \n", sigma)
    print("\n\n")
    print("Updated sigma values: \n", updated_sigma)
    print("\n\n")
    print("2E(E.T)E - tr(E(E.T))E \n", constraint_updated_E)
    print("\n\n")
    print("Translation vector: \n", t)
    print("\n\n")
    print("Rotation matrix: \n", R)

if __name__ == "__main__":
    main()