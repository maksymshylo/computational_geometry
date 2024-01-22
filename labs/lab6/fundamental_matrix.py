import argparse

import cv2
import numpy as np
from numpy import ndarray, trace
from numpy.linalg import det, inv
from numpy.linalg import matrix_rank as rank
from scipy.linalg import null_space


def get_matches(img_left: ndarray, img_right: ndarray) -> tuple[ndarray, ndarray]:
    """
    Run cv2.SIFT with cv2.BFMatcher.
    Calculate matches.

    Args:
        img_left: Left image. (Source image).
                 Shape (Any, Any).
        img_right: Right image. (Destination image).
                 Shape (Any, Any).

    Returns:
        A tuple containing left and right points.
        Each ndarray has a shape of (Any, 1, 2).
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_left, None)
    kp2, des2 = sift.detectAndCompute(img_right, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # take only good matches
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    left_pts = np.array([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    right_pts = np.array([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return left_pts, right_pts


def convert_to_homogeneous(
    left_pts: ndarray, right_pts: ndarray
) -> tuple[ndarray, ndarray]:
    """
    Add ones to make all coordinates homogeneous

    Args:
        left_pts: Points on the left image.
                 Shape: (Any, 1, 2).
        right_pts: Points on the right image.
                 Shape: (Any, 1, 2).

    Returns:
        A tuple containing homogeneous points on the left and right images.
        Each ndarray has a shape of (Any, 1, 3).
    """
    ones_arr = np.ones((left_pts.shape[0], 1))

    left_pts_h = np.dstack((left_pts, ones_arr))
    right_pts_h = np.dstack((right_pts, ones_arr))

    return left_pts_h, right_pts_h


def get_7pts(left_pts: ndarray, right_pts: ndarray) -> ndarray:
    """
    Get random 7 points from left_pts, right_pts points.

    Args:
        left_pts: Points on the left image. (homogeneous).
                 Shape: (Any, 1, 3).
        right_pts: Points on the right image. (homogeneous).
                 Shape: (Any, 1, 3).

    Returns:
        Random 7 points. Shape: (7, 9).
    """
    random_indices = np.random.randint(0, len(left_pts), 7)
    X = np.matmul(
        right_pts[random_indices].reshape(-1, 3, 1), left_pts[random_indices]
    ).reshape(7, 9)

    return X


def apply_fundamental_matrix(
    left_pts: ndarray,
    right_pts: ndarray,
    F: ndarray,
) -> ndarray:
    """
    Get inliers points by applying fundamental matrix to the points.
    Formula  sum_(|x`Fx|/norm_coefficient)

    Args:
        left_pts: Points on the left image. (homogeneous).
                 Shape: (Any, 1, 3).
        right_pts: Points on the right image. (homogeneous).
                 Shape: (Any, 1, 3).
        F: Fundamental matrix.
           Shape: (3, 3).

    Returns:
        Inliers array.
        Shape: (Any, ).
    """
    x_prime_F = np.matmul(right_pts, F)
    norm_coefficient = np.sqrt(
        x_prime_F[:, 0][:, 0] ** 2 + x_prime_F[:, 0][:, 1] ** 2
    ).ravel()
    inliers = abs(x_prime_F @ (left_pts.reshape(-1, 3, 1))).ravel()
    inliers = inliers / norm_coefficient

    return inliers


def get_best_fundamental_matrix(
    left_pts: ndarray,
    right_pts: ndarray,
    epsilon: int,
    n_iter: int,
) -> tuple[ndarray, ndarray, int]:
    """
    Calculate the best fundamental matrix.
    Use number of inliers as an accuracy criterion.

    Args:
        left_pts: Points on the left image. (homogeneous).
                 Shape: (Any, 1, 3).
        right_pts: Points on the right image. (homogeneous).
                 Shape: (Any, 1, 3).
        epsilon: Threshold parameter in pixels.
        n_iter: Number of iterations.

    Returns:
        A tuple:
            The best fundamental matrix. Shape: (3, 3).
            Inliers array. Shape: (Any, ).
            Number of inliers (the best accuracy).
    """
    best_accuracy = 0
    for _ in range(n_iter):
        # get solution of X system
        X = get_7pts(left_pts, right_pts)
        ker_X = null_space(X)
        F1, F2 = ker_X[:, 0].reshape(3, 3), ker_X[:, 1].reshape(3, 3)

        if rank(F1) == 2:
            inliers = apply_fundamental_matrix(left_pts, right_pts, F1)
            accuracy = np.sum(inliers < epsilon)
            if accuracy > best_accuracy:
                F_best = F1
                best_accuracy = accuracy
                best_inliers = inliers

        elif rank(F2) == 2:
            inliers = apply_fundamental_matrix(left_pts, right_pts, F2)
            accuracy = np.sum(inliers < epsilon)
            if accuracy > best_accuracy:
                F_best = F1
                best_accuracy = accuracy
                best_inliers = inliers

        elif rank(F1) == 3 and rank(F2) == 3:
            # roots for polynomial equation
            p = np.array(
                [
                    det(F1),
                    det(F1) * trace(F2 @ inv(F1)),
                    det(F2) * trace(F1 @ inv(F2)),
                    det(F2),
                ]
            )
            roots = np.roots(p)
            # take only real roots
            real_roots = roots[~np.iscomplex(roots)].real
            for real_root in real_roots:
                F = real_root * F1 + F2
                if rank(F) == 2:
                    inliers = apply_fundamental_matrix(left_pts, right_pts, F)
                    accuracy = np.sum(inliers < epsilon)
                    if accuracy > best_accuracy:
                        F_best = F
                        best_accuracy = accuracy
                        best_inliers = inliers

    return F_best, best_inliers, best_accuracy


def calculate_epipolar_lines(points: ndarray, F: ndarray, whichImage: str) -> ndarray:
    """
    Calculate epipolar lines by applying fundamental matrix to points.

    Args:
        points: Points on the left or right image.
                 Shape: (Any, 1, 2).
        F: Fundamental matrix.
           Shape: (3, 3).
        whichImage: "left" or "right".

    Returns:
        Lines array.
        Shape: (Any, 3).
    """
    assert whichImage in [
        "left",
        "right",
    ], 'whichImage argument must be either "left" or "right"'

    ones_arr = np.ones((points.shape[0], 1))
    points = np.dstack((points, ones_arr))

    if whichImage == "left":
        lines = np.matmul(F, points.reshape(-1, 3, 1)).reshape(-1, 3)

    if whichImage == "right":
        lines = np.matmul(F.T, points.reshape(-1, 3, 1)).reshape(-1, 3)

    return lines


def draw_lines_and_points(img, lines, points):
    """
    Draw lines and points on the image.

    Args:
        img: Shape:  (Any, Any).
        lines: Corresponding epipolar lines.
        points: Points for the image.

    Returns:
        RGB image with lines. Shape: (Any, Any, 3).
    """

    r, c = img.shape
    img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for r, pt in zip(lines, points):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        img_with_lines = cv2.line(img_with_lines, (x0, y0), (x1, y1), color, 1)
        img_with_lines = cv2.circle(
            img_with_lines, tuple(pt.ravel().astype(np.int64)), 5, color, -1
        )

    return img_with_lines


def draw_epipolar_lines(
    left_pts: ndarray,
    right_pts: ndarray,
    img_left: ndarray,
    img_right: ndarray,
    F: ndarray,
    inliers: ndarray,
    epsilon: int,
) -> tuple[ndarray, ndarray]:
    """
    Draw epipolar lines from the right image
    and inliers from the left image
    on the left image.

    Draw epipolar lines from the left image
    and inliers from the right image
    on the right image.

    Args:
        left_pts: Points on the left image.
                 Shape: (Any, 1, 2).
        right_pts: Points on the right image.
                 Shape: (Any, 1, 2).
        img_left: Left image.
                 Shape: (Any, Any).
        img_right: Right image.
                 Shape: (Any, Any).
        inliers: Inliers array.
                 Shape: (Any, ).
        epsilon: Threshold parameter in pixels.

    Returns:
        A tuple containing left and right images with epipolar lines.
        Each ndarray has a shape of (Any, Any, 3).
    """
    mask = inliers < epsilon

    left_inliers = left_pts[mask == 1]
    right_inliers = right_pts[mask == 1]

    right_lines_on_the_left = calculate_epipolar_lines(
        points=right_inliers, F=F, whichImage="right"
    )
    left_img_with_epiliens = draw_lines_and_points(
        img=img_left, lines=right_lines_on_the_left, points=left_inliers
    )

    left_lines_on_the_right = calculate_epipolar_lines(
        points=left_inliers, F=F, whichImage="left"
    )
    right_img_with_epiliens = draw_lines_and_points(
        img=img_right, lines=left_lines_on_the_right, points=right_inliers
    )

    return left_img_with_epiliens, right_img_with_epiliens
