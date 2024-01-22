import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numpy import ndarray
from PIL import Image


def calc_disparity_vectors(x_range: tuple, y_range: tuple) -> ndarray:
    """
    Calculate disparity vectors.

    Args:
        x_range: Disparity of X axis
        y_range: Disparity of Y axis
    Returns
        Disparity vectors
    """
    mindx, maxdx = x_range
    mindy, maxdy = y_range

    disp_x = np.arange(mindx, maxdx + 1, 1)
    disp_y = np.arange(mindy, maxdy + 1, 1)

    x2d, y2d = np.meshgrid(disp_y, disp_x)
    disp_vectors = np.column_stack((x2d.ravel(), y2d.ravel()))

    return disp_vectors


@njit(fastmath=True, nogil=True)
def calc_unary_penalties(
    left_image: ndarray, right_image: ndarray, disp_vectors: ndarray
) -> ndarray:
    """
    Calculate unary penalties.

    Args:
        left_image: Left image
        right_image: Right image
        disp_vectors: Disparity vectors
    Returns
        Unary penalties
    """
    height, width = left_image.shape
    n_labels = disp_vectors.shape[0]
    unary_p = np.full((height, width, n_labels), -np.inf)
    for i in range(height):
        for j in range(width):
            for index, d in enumerate(disp_vectors):
                # fill only pixels with possible disparity
                if 0 <= i - d[0] < height and 0 <= j - d[1] < width:
                    unary_p[i, j, index] = -abs(
                        left_image[i, j] - right_image[i - d[0], j - d[1]]
                    )

    return unary_p


def calc_binary_penalties(smoothing_coef: float, disp_vectors: ndarray) -> ndarray:
    """
    Calculate binary penalties with applying smooth coefficient.

    Args:
        smoothing_coef: Smoothing coefficient
        disp_vectors: Disparity vectors
    Returns
        Binary penalties
    """
    n_labels = disp_vectors.shape[0]
    binary_p = np.full((n_labels, n_labels), -1)
    for i in range(n_labels):
        for j in range(n_labels):
            disp_vec_x = disp_vectors[i]
            disp_vec_y = disp_vectors[j]
            # calculate norm of difference
            binary_p[i, j] = np.linalg.norm(disp_vec_x - disp_vec_y)

    return -smoothing_coef * binary_p


@njit(fastmath=True, cache=True)
def init_best_wp(
    height: int,
    width: int,
    n_labels: int,
    unary_p: ndarray,
    binary_p: ndarray,
    best_wp: ndarray,
) -> ndarray:
    """
    Update best weights paths for Right and Down directions.

    Args:
        height: image height
        width: image width
        n_labels: number of labels
        unary_p: unary penalties
        binary_p: binary penalties
        best_wp: best weights paths for each direction (Left,Right,Up,Down)
    Returns
        Updated best weights paths
    """
    # for each pixel of input channel
    # going from bottom-right to top-left pixel
    for i in range(height - 2, -1, -1):
        for j in range(width - 2, -1, -1):
            # for each label in pixel
            for k in range(n_labels):
                # best_wp[i,j,1,k] - Right direction
                # best_wp[i,j,3,k] - Down direction
                # calculate the best path weight according to formula
                best_wp[i, j, 3, k] = max(
                    best_wp[i + 1, j, 3, :] + unary_p[i + 1, j, :] + binary_p[k, :]
                )
                best_wp[i, j, 1, k] = max(
                    best_wp[i, j + 1, 1, :] + unary_p[i, j + 1, :] + binary_p[k, :]
                )

    return best_wp


@njit(fastmath=True, cache=True)
def forward_pass(
    height: int,
    width: int,
    n_labels: int,
    unary_p: ndarray,
    binary_p: ndarray,
    best_wp: ndarray,
) -> ndarray:
    """
    Update best weights paths for Left and Up directions.

    Parameters
        height: image height
        width: image width
        n_labels: number of labels
        unary_p: unary penalties
        binary_p: binary penalties
        best_wp: best weights paths for each direction (Left,Right,Up,Down)
    Returns
        Updated best weights paths
    """
    # for each pixel of input channel
    for i in range(1, height):
        for j in range(1, width):
            # for each label in pixel
            for k in range(n_labels):
                # P[i,j,0,k] - Left direction
                # P[i,j,2,k] - Up direction
                # calculate the best path weight according to formula
                best_wp[i, j, 0, k] = max(
                    best_wp[i, j - 1, 0, :] + unary_p[i, j - 1, :] + binary_p[:, k]
                )
                best_wp[i, j, 2, k] = max(
                    best_wp[i - 1, j, 2, :] + unary_p[i - 1, j, :] + binary_p[:, k]
                )

    return best_wp


def semi_global_matching(
    left_image: ndarray,
    right_image: ndarray,
    smoothing_coef: float,
    disp_vectors: ndarray,
) -> ndarray:
    """
    Run Semi Global Matching algorithm.

    Args:
        left_image: Left image
        right_image: Right image
        smoothing_coef: Smoothing coefficient for binary penalties
        disp_vectors: Disparity vectors
    Returns
        Optimal labelling
    """
    n_labels = disp_vectors.shape[0]
    height, width = left_image.shape

    unary_p = calc_unary_penalties(left_image, right_image, disp_vectors)
    binary_p = calc_binary_penalties(smoothing_coef, disp_vectors)

    # fill best weights paths
    best_wp = np.zeros((height, width, 4, n_labels))
    best_wp = init_best_wp(height, width, n_labels, unary_p, binary_p, best_wp)
    best_wp = forward_pass(height, width, n_labels, unary_p, binary_p, best_wp)

    sum_over_directions = (
        best_wp[:, :, 0, :]
        + best_wp[:, :, 1, :]
        + best_wp[:, :, 2, :]
        + best_wp[:, :, 3, :]
        + unary_p
    )

    optimal_labelling = np.argmax(sum_over_directions, axis=2)

    return optimal_labelling


def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--left_image", type=str, required=True, help="Path to left image."
    )
    parser.add_argument(
        "--right_image", type=str, required=True, help="Path to right image."
    )
    parser.add_argument(
        "--smoothing_coef",
        type=float,
        required=True,
        help="Smoothing coefficient for binary penalties.",
    )
    parser.add_argument(
        "--min_dx", type=int, required=True, help="Min value for horizontal disparity."
    )
    parser.add_argument(
        "--max_dx", type=int, required=True, help="Max value for horizontal disparity."
    )
    parser.add_argument(
        "--min_dy", type=int, required=True, help="Min value for vertical disparity."
    )
    parser.add_argument(
        "--max_dy", type=int, required=True, help="Max value for vertical disparity."
    )
    args = parser.parse_args()

    left_image = np.array(Image.open(args.left_image).convert("L")).astype(float)
    right_image = np.array(Image.open(args.right_image).convert("L")).astype(float)

    x_range = (args.min_dx, args.max_dx)
    y_range = (args.min_dy, args.max_dy)

    # perform sgm and get optimal labelling
    start_time = time.time()

    disp_vectors = calc_disparity_vectors(x_range, y_range)
    optimal_labelling = semi_global_matching(
        left_image, right_image, args.smoothing_coef, disp_vectors
    )

    end_time = time.time()

    print("total time: ", np.round(end_time - start_time, 4), "sec")

    output_image = np.linalg.norm(disp_vectors[optimal_labelling], axis=2)

    plt.imsave("disparity_map.png", output_image, cmap="gray")

    # plt.figure(figsize=(20, 20))
    # plt.axis('off')
    # plt.imshow(bgr)
    # plt.show()


if __name__ == "__main__":
    main()
