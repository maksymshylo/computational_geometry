import time
import argparse
import numpy as np
import cv2
from numba import njit
import matplotlib.pyplot as plt
from PIL import Image


def set_params(x_range, y_range):
    '''
    Parameters
        x_range: tuple
            disparity of X axis
        y_range: tuple
            disparity of Y axis
    Returns
        disparity_vec: ndarray
            array of disparity vectors
    calculates disparity_vec
    '''
    mindx, maxdx = x_range
    mindy, maxdy = y_range
    disp_x = np.arange(mindx, maxdx+1, 1)
    disp_y = np.arange(mindy, maxdy+1, 1)
    X2D, Y2D = np.meshgrid(disp_y, disp_x)
    disparity_vec = np.column_stack((X2D.ravel(), Y2D.ravel()))

    return disparity_vec


@njit(fastmath=True, nogil=True)
def calculate_q(left, right, disparity_vec):
    '''
    Parameters
        left: ndarray
            left image
        right: ndarray
            right image
    Returns
        Q: ndarray
            unary penalties
    calculates unary penalties
    '''
    height, width = left.shape
    n_labels = disparity_vec.shape[0]
    Q = np.full((height, width, n_labels), -np.inf)
    for i in range(height):
        for j in range(width):
            for index, d in enumerate(disparity_vec):
                # fill only pixels with possible disparity
                if 0 <= i-d[0] < height and 0 <= j-d[1] < width:
                    Q[i, j, index] = -abs(left[i, j] - right[i-d[0], j-d[1]])
    return Q


def calc_g(smooth_coef, mapping):
    '''
    Parameters
        smooth_coef: float
            smoothing coefficient
        mapping: dissparit
            array of disparity vectors
    Returns
        g: ndarray
            binary penalties
    calculates binary penalties
    '''
    n_labels = mapping.shape[0]
    g = np.full((n_labels, n_labels), -1)
    for i in range(n_labels):
        for j in range(n_labels):
            vec1 = mapping[i]
            vec2 = mapping[j]
            # calculate norm of difference
            g[i, j] = np.linalg.norm(vec1-vec2)
    return -smooth_coef*g


@njit(fastmath=True, cache=True)
def init(height, width, n_labels, Q, g, P):
    '''
    Parameters
        height: int
            height of input image
        width: int
            width of input image
        n_labels: int
            number of labels in labelset
        Q: ndarray
            array of unary penalties
        g: ndarray
            array of binary penalties
        P: ndarray
            array consist of best path weight for each direction
                (Left,Right,Up,Down)
    Returns
        P: ndarray
            updated P
    updates P
    '''
    # for each pixel of input channel
    # going from bottom-right to top-left pixel
    for i in range( height-2,-1, -1):
        for j in range( width-2,-1, -1):
            # for each label in pixel
            for k in range(n_labels):
                # P[i,j,1,k] - Right direction
                # P[i,j,3,k] - Down direction
                # calculate best path weight according to formula
                P[i, j, 3, k] = max(P[i+1, j, 3, :] + Q[i+1, j, :] + g[k, :])
                P[i, j, 1, k] = max(P[i, j+1, 1, :] + Q[i, j+1, :] + g[k, :])
    return P

@njit(fastmath=True, cache=True)
def forward_pass(height, width, n_labels, Q, g, P):
    '''
    Parameters
        height: int
            height of input image 
        width: int
            width of input image 
        n_labels: int
            number of labels in labelset
        Q: ndarray
            array of unary penalties
        g: ndarray
            array of binary penalties
        P: ndarray
            array consist of best path weight for each direction (Left,Right,Up,Down)
    Returns
        P: ndarray
            array consist of best path weight for each direction (Left,Right,Up,Down)
    updates P
    '''
    # for each pixel of input channel
    for i in range(1, height):
        for j in range(1, width):
            # for each label in pixel
            for k in range(n_labels):
                # P[i,j,0,k] - Left direction
                # P[i,j,2,k] - Up direction
                # calculate best path weight according to formula
                P[i, j, 0, k] = max(P[i, j-1, 0, :] + Q[i, j-1, :] + g[:, k])
                P[i, j, 2, k] = max(P[i-1, j, 2, :] + Q[i-1, j, :] + g[:, k])

    return P



def sgm(imleft,imright,smooth,disparity_vec):
    '''
    Parameters
        imleft: ndarray
            left image
        imright: ndarray
            right image
        smooth: float
            smoothing coefficient for binary penalties
        disparity_vec: ndarray
            array of disparity vectors
    Returns
        optimal_labelling: ndarray
            optimal labelling after SGM algorithm
    performs SGM algorithm for given penalties
    '''
    n_labels = disparity_vec.shape[0]
    height,width = imleft.shape

    Q = calculate_q(imleft, imright, disparity_vec)
    g = calc_g(smooth, disparity_vec)

    P = np.zeros((height, width, 4, n_labels))
    P = init(height, width, n_labels, Q, g, P)
    P = forward_pass(height, width, n_labels, Q, g, P)

    optimal_labelling = np.argmax(
        P[:, :, 0, :] + P[:, :, 1, :] + P[:, :, 2, :] + P[:, :, 3, :] + Q, axis=2)
    return optimal_labelling





def cart2pol(x, y):
    '''
    Parameters
        x: int
            x coordinate in cartesian notion
        y: int 
            y coordinate in cartesian notion
    Returns
        theta: float
            angle coordinate in polar system
        rho: float
            norm of (x,y) vector
    transform cartesian coordinates to polar
    '''
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho



def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("left_image", type=str, help="path to left image")
    parser.add_argument("right_image", type=str, help="path to right image")
    parser.add_argument("smoothing_coef", type=float, help="smoothing coefficient for binary penalties")
    parser.add_argument("min_dx", type=int, help="min value for horizontal disparity")
    parser.add_argument("max_dx", type=int, help="max value for horizontal disparity")
    parser.add_argument("min_dy", type=int, help="min value for vertical disparity")
    parser.add_argument("max_dy", type=int, help="max value for vertical disparity")
    args = parser.parse_args()


    imleft = np.array(Image.open(args.left_image).convert('L')).astype(float)
    imright = np.array(Image.open(args.right_image).convert('L')).astype(float)

    x_range = (args.min_dx, args.max_dx)
    y_range = (args.min_dy, args.max_dy)
    
    # perform sgm and get optimal labelling
    a = time.time()
    disparity_vec = set_params(x_range, y_range)
    optimal_labelling = sgm(imleft,imright,args.smoothing_coef,disparity_vec)
    print('total time: ', np.round(time.time() - a,4), 'sec')

    
    output_image = np.linalg.norm(disparity_vec[optimal_labelling],axis=2)


    plt.imsave('disparity_map.png',output_image,cmap='gray')

    #plt.figure(figsize=(20, 20))
    #plt.axis('off')
    #plt.imshow(bgr)
    #plt.show()      



if __name__ == "__main__":
    main()
