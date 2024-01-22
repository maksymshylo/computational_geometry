import argparse

import cv2
import numpy as np
from numpy import ndarray


def get_matches(src_img: ndarray, dst_img: ndarray) -> tuple[ndarray, ndarray]:
    """
    Run cv2.SIFT with cv2.FlannBasedMatcher.
    Calculate matches.

    Args:
        src_img: Source image
        dst_img: Destination image

    Returns:
        Source and destination points
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(src_img, None)
    kp2, des2 = sift.detectAndCompute(dst_img, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # get matches
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    # sort
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return src_pts, dst_pts


def interpolate(image: ndarray, x: int, y: int) -> ndarray:
    """
    Interpolate image points.

    Args:
        image: Input image
        x: x point
        y: y point
    Returns:
        Weighted sum of 4 nearest neighbours
    """
    # all neighbours coordinates
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1

    x0, x1 = np.clip(x0, 0, image.shape[1] - 1), np.clip(x1, 0, image.shape[1] - 1)
    y0, y1 = np.clip(y0, 0, image.shape[0] - 1), np.clip(y1, 0, image.shape[0] - 1)

    # weighted sum of 4 nearest neighbours
    weighted_sum = (
        (image[y0, x0].T * (x1 - x) * (y1 - y)).T
        + (image[y1, x0].T * (x1 - x) * (y - y0)).T
        + (image[y0, x1].T * (x - x0) * (y1 - y)).T
        + (image[y1, x1].T * (x - x0) * (y - y0)).T
    )

    return weighted_sum


def apply_homography(
    src_img: ndarray, H: ndarray, img_size: tuple[int, int]
) -> ndarray:
    """
    Apply homography to src_img.

    Args:
        src_img: Input image
        H: Homography matrix
        img_size: height and width
    Returns:
        Warped image.
    """
    h, w = img_size
    # inverse homography matrix to apply it to output image
    H = np.linalg.inv(H)
    coords = np.zeros((h, w, 2))
    for i in range(h):
        for j in range(w):
            unnorm = H @ np.array([j, i, 1])
            coords[i, j, :] = (unnorm / unnorm[-1])[:-1]

    # new coordinates after homography
    coords = coords.reshape(-1, 2)
    # interpolate colors
    warped_img = interpolate(src_img, coords[:, 0], coords[:, 1])
    warped_img = warped_img.reshape(h, w, 3)
    warped_img = warped_img.astype(np.uint8)
    return warped_img


def calculate_homography(p1: ndarray, p2: ndarray) -> ndarray:
    """
    Calculate homography matrix from 4 correspondent points.

    Args:
        p1: x,y point
        p2: x,y point
    Returns:
        Homography matrix
    """
    A = np.array(
        [
            [
                p1[0][0][0],
                p1[0][0][1],
                1,
                0,
                0,
                0,
                -p2[0][0][0] * p1[0][0][0],
                -p2[0][0][0] * p1[0][0][1],
                -p2[0][0][0],
            ],
            [
                0,
                0,
                0,
                p1[0][0][0],
                p1[0][0][1],
                1,
                -p2[0][0][1] * p1[0][0][0],
                -p2[0][0][1] * p1[0][0][1],
                -p2[0][0][1],
            ],
            [
                p1[1][0][0],
                p1[1][0][1],
                1,
                0,
                0,
                0,
                -p2[1][0][0] * p1[1][0][0],
                -p2[1][0][0] * p1[1][0][1],
                -p2[1][0][0],
            ],
            [
                0,
                0,
                0,
                p1[1][0][0],
                p1[1][0][1],
                1,
                -p2[1][0][1] * p1[1][0][0],
                -p2[1][0][1] * p1[1][0][1],
                -p2[1][0][1],
            ],
            [
                p1[2][0][0],
                p1[2][0][1],
                1,
                0,
                0,
                0,
                -p2[2][0][0] * p1[2][0][0],
                -p2[2][0][0] * p1[2][0][1],
                -p2[2][0][0],
            ],
            [
                0,
                0,
                0,
                p1[2][0][0],
                p1[2][0][1],
                1,
                -p2[2][0][1] * p1[2][0][0],
                -p2[2][0][1] * p1[2][0][1],
                -p2[2][0][1],
            ],
            [
                p1[3][0][0],
                p1[3][0][1],
                1,
                0,
                0,
                0,
                -p2[3][0][0] * p1[3][0][0],
                -p2[3][0][0] * p1[3][0][1],
                -p2[3][0][0],
            ],
            [
                0,
                0,
                0,
                p1[3][0][0],
                p1[3][0][1],
                1,
                -p2[3][0][1] * p1[3][0][0],
                -p2[3][0][1] * p1[3][0][1],
                -p2[3][0][1],
            ],
        ]
    )

    _, _, Vh = np.linalg.svd(A)
    # last element = 1
    H = (Vh[-1, :] / Vh[-1, -1]).reshape(3, 3)
    return H


def ransac(
    src_pts: ndarray,
    dst_pts: ndarray,
    ransacReprojThreshold: int = 5,
    maxIters: int = 2000,
) -> ndarray:
    """
    Run ransac algorithm.

    Args:
        src_pts: Source points
        dst_pts: Destination points
    Returns:
        Best homography matrix.
    """
    # implementation of ransac algorithm for finding the best homography matrix

    # adding ones array to imitate 3D
    src_pts = np.dstack((src_pts, np.ones((src_pts.shape[0], src_pts.shape[1]))))
    dst_pts = np.dstack((dst_pts, np.ones((dst_pts.shape[0], dst_pts.shape[1]))))

    best_count_matches = 0
    for iteration in range(maxIters):
        # get 4 random points
        random_indices = np.random.randint(0, len(src_pts) - 1, 4)
        random_kp_src, random_kp_dst = src_pts[random_indices], dst_pts[random_indices]

        # calculate a homography
        H_current = calculate_homography(random_kp_src, random_kp_dst)
        count_matches = 0

        # check all points
        for i in range(len(src_pts)):
            unnorm = H_current @ src_pts[i][0]
            normalized = unnorm / unnorm[-1]
            # calculate norm
            norm = np.linalg.norm(normalized - dst_pts[i][0])
            # check solution
            if norm < ransacReprojThreshold:
                count_matches += 1

        if count_matches >= best_count_matches:
            # update best homography if better match found
            best_count_matches = count_matches
            H_best = H_current
    return H_best


def perspective_transform(points: list, H: ndarray) -> ndarray:
    """
    Apply homography transformation matrix to points.

    Args:
        points: Points to apply homography
        H: Homography transformation matrix
    Returns:
        New points.
    """
    new_points = []
    for point in points:
        p = H @ np.array([point[0][0], point[0][1], 1])
        p = p / p[-1]
        new_points.append(p[:-1])
    new_points = np.array(new_points, dtype=np.float32).reshape(points.shape)
    return new_points


def crop(image: ndarray) -> ndarray:
    """
    Remove black border.

    Args:
        image: Input image
    Returns:
        Cropped image.
    """
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    cropped_img = image[
        np.min(y_nonzero) : np.max(y_nonzero), np.min(x_nonzero) : np.max(x_nonzero)
    ]
    return cropped_img


def stitch_images(img1: ndarray, img2: ndarray, H: ndarray) -> ndarray:
    """
    Stitch 2 images using homography matrix.

    Args:
        img1: Input image
        img2: Input image
        H: Homography transformation matrix
    Returns:
        Stitched image.
    """

    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape
    # edge points from image 1
    list_points_1 = np.float32(
        [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]
    ).reshape(-1, 1, 2)
    # edge points from image 2
    list_points_2 = np.float32(
        [[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]
    ).reshape(-1, 1, 2)

    # match points
    matched_points_12 = perspective_transform(list_points_2, H)
    matches = np.vstack((list_points_1, matched_points_12))

    # new edges coordinate
    x_min, y_min = np.int32(matches.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(matches.max(axis=0).ravel() + 0.5)

    # translation matrix
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    # stitch 2 images
    output_img = apply_homography(
        img2, H_translation @ H, (y_max - y_min, x_max - x_min)
    )
    output_img[-y_min : rows1 - y_min, -x_min : cols1 - x_min] = img1
    # crop black edges
    output_img = crop(output_img)
    return output_img


def main():
    # Run stitching process
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, required=True, help="path to img1")
    parser.add_argument("--img2", type=str, required=True, help="path to img2")
    parser.add_argument(
        "--ransac_iter",
        type=int,
        required=True,
        help="max iteration for ransac algorithm",
    )
    parser.add_argument(
        "--out_img", type=str, required=True, help="path to output image"
    )
    args = parser.parse_args()

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    src_pts, dst_pts = get_matches(img1, img2)

    h_best = ransac(
        src_pts, dst_pts, ransacReprojThreshold=3, maxIters=args.ransac_iter
    )

    panorama = stitch_images(img2, img1, h_best)

    # save output image
    cv2.imwrite(args.out_img, panorama)


if __name__ == "__main__":
    main()
