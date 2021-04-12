import argparse
import numpy as np
import cv2
from scipy.linalg import null_space
from numpy.linalg import det, inv, matrix_rank as rank
from numpy import trace


def get_matches(src, dst):
    """
    calculete matches points from src and dst images
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(src,None)
    kp2, des2 = sift.detectAndCompute(dst,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # take only good matches
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    src_pts = np.array([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.array([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    return src_pts, dst_pts


def get_7pts(src_pts,dst_pts):
    """
    create matrix (7,9) of random matched pairs
    from src_pts, dst_pts points
    """
    random_indices = np.random.randint(0,len(src_pts),7)
    X = np.matmul(  dst_pts[random_indices].reshape(-1,3,1),
                    src_pts[random_indices]
                 ).reshape(7,9)
    return X


def get_accuracy(src_pts, dst_pts, F, epsilon):
    """
    calculate number of inliers points for current F
    formula  sum_(|x`Fx|/norm_coefficient)
    """

    x_prime_F = np.matmul(dst_pts,F)
    norm_coefficient = np.sqrt(x_prime_F[:,0][:,0]**2 + x_prime_F[:,0][:,1]**2).ravel()
    inliers = abs(x_prime_F@(src_pts.reshape(-1,3,1))).ravel()
    inliers = inliers/norm_coefficient
    accuracy = np.sum(inliers<epsilon)
    return accuracy


def get_mask(src_pts, dst_pts,F,epsilon):
    """
    calculate inliers points for current F
    formula |x`Fx|/norm_coefficient
    """
    x_prime_F = np.matmul(dst_pts,F)
    norm_coefficient = np.sqrt(x_prime_F[:,0][:,0]**2 + x_prime_F[:,0][:,1]**2).ravel()
    inliers = abs(x_prime_F@(src_pts.reshape(-1,3,1))).ravel()
    inliers = inliers/norm_coefficient
    mask = (inliers<epsilon)
    return mask


def calculate_fundamental(src_pts, dst_pts, epsilon, n_iter):
    best_accuracy = 0
    for _ in range(n_iter):
        # get solution of X system
        X = get_7pts(src_pts,dst_pts)
        ker_X = null_space(X)
        F1, F2 = ker_X[:,0].reshape(3,3), ker_X[:,1].reshape(3,3)

        if rank(F1) == 2:
            accuracy = get_accuracy(src_pts, dst_pts,F1,epsilon)
            if accuracy > best_accuracy:
                F_best = F1
                best_accuracy = accuracy

        elif rank(F2) == 2:
            accuracy = get_accuracy(src_pts, dst_pts,F2,epsilon)
            if accuracy > best_accuracy:
                F_best = F1
                best_accuracy = accuracy

        elif rank(F1) == 3 and rank(F2) == 3:
            # roots for polynomial equation
            p = np.array([det(F1), det(F1)*trace(F2@inv(F1)),
                          det(F2)*trace(F1@inv(F2)), det(F2)
                         ])
            roots = np.roots(p)
            # take only real roots
            roots = roots[~np.iscomplex(roots)].real
            for root in roots:
                F = (root*F1 + F2)
                if rank(F) == 2:
                    accuracy = get_accuracy(src_pts, dst_pts,F,epsilon)
                    if accuracy > best_accuracy:
                        F_best = F
                        best_accuracy = accuracy

    return F_best, best_accuracy


def drawlines(img_left, img_right, lines, pts1,pts2):
    """
    img_left - image on which we draw the epilines for the points in img_right
    lines - corresponding epilines 
    """

    r,c = img_left.shape
    img_left = cv2.cvtColor(img_left,cv2.COLOR_GRAY2BGR)
    img_right = cv2.cvtColor(img_right,cv2.COLOR_GRAY2BGR)

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())

        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])

        img_left = cv2.line(img_left, (x0,y0), (x1,y1), color,1)
        img_left = cv2.circle(img_left,tuple(pt1.ravel().astype(np.int64)),5,color,-1)
        img_right = cv2.circle(img_right,tuple(pt2.ravel().astype(np.int64)),5,color,-1)

    return img_left,img_right


def computeEpilines(points, whichImage, F):
    """
    calculate epipolar lines
    whichImage = 1 for left image
    whichImage = 2 for right image
    """

    ones_arr_points = np.ones((points.shape[0],1))
    points = np.dstack((points,ones_arr_points))

    if whichImage == 1:
        lines2 = np.matmul(F, points.reshape(-1,3,1)).reshape(-1,3)

        return lines2

    if whichImage == 2:
        lines1 = np.matmul(F.T, points.reshape(-1,3,1)).reshape(-1,3)

        return lines1



def get_epilines_images(src_pts, dst_pts,img_left, img_right, mask, F):

    pts1 = src_pts[mask==1]
    pts2 = dst_pts[mask==1]

    lines1 = computeEpilines(pts2, 2, F)
    epilines1,_ = drawlines(img_left, img_right, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = computeEpilines(pts1, 1, F)
    epilines2,_ = drawlines(img_right, img_left, lines2, pts2, pts1)

    return epilines1, epilines2



def main():

    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("left_image",  type=str,   help="left input image")
    parser.add_argument("right_image", type=str,   help="right input image")
    parser.add_argument("epsilon",     type=int,   help="threshold parameter in PIXELS")
    parser.add_argument("n_iter",      type=int,   help="number of iterations")

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

    F_best, best_accuracy = calculate_fundamental(src_pts, dst_pts, args.epsilon, args.n_iter)
    mask = get_mask(src_pts, dst_pts, F_best, args.epsilon)

    epilines1, epilines2 = get_epilines_images(src_pts_vis, dst_pts_vis, img_left, img_right, mask, F_best)

    cv2.imwrite("epilines1.png", epilines1)
    cv2.imwrite("epilines2.png", epilines2)

if __name__ == "__main__":
    main()