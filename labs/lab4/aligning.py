import cv2
import numpy as np
from numpy import ndarray
from numpy.linalg import norm


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


def interpolate(im: ndarray, x: int, y: int) -> ndarray:
    """
    Interpolate image points.

    Args:
        im: Input image
        x: x point
        y: y point
    Returns:
        Weighted sum of 4 nearest neighbours
    """
    # all neighbours coordinates
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T


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
        Warped image
    """
    h, w = img_size
    # inverse homography matrix to apply it to output image
    H = np.linalg.inv(H)
    coords = np.zeros((h, w, 2))
    for i in range(h):
        for j in range(w):
            unnorm = H @ np.array([j, i, 1])
            coords[i, j, :] = (unnorm / unnorm[-1])[:-1]
    # new coordimates after homography
    coords = coords.reshape(-1, 2)
    # interpolate colors
    warped = interpolate(src_img, coords[:, 0], coords[:, 1])
    warped = warped.reshape(h, w, 3)
    warped = warped.astype(np.uint8)
    return warped


def calculate_homography(p1, p2):
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
    maxIters: int = 3000,
) -> ndarray:
    """
    Run ransac algorithm. Find the best homography.

    Args:
        src_pts: Source points
        dst_pts: Destination points
        ransacReprojThreshold: Threshold to count matches
        maxIters: Number of RANSAC iterations
    Returns:
        Best homography matrix
    """
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
            # update the best homography if better match found
            best_count_matches = count_matches
            H_best = H_current

    return H_best


def perspective_transform(points: ndarray, homography_matrix: ndarray) -> ndarray:
    """
    Apply homography transformation matrix to points.

    Args:
        points: Points to apply homography
        homography_matrix: Homography transformation matrix
    Returns:
        New points.
    """
    new_points = []
    for point in points:
        p = homography_matrix @ np.array([point[0][0], point[0][1], 1])
        p = p / p[-1]
        new_points.append(p[:-1])
    new_points = np.array(new_points, dtype=np.float32).reshape(points.shape)
    return new_points


def align_images(input_images: list) -> list:
    """
    Align images. First image in the input_images is the source image.

    Args:
        input_images: Images to align

    Returns:
        List of aligned images. Each element is ndarray
    """
    number_of_images = len(input_images)
    base_image = input_images[0]

    edge_points = np.array(
        [
            [0, 0],
            [0, base_image.shape[0] - 1],
            [base_image.shape[1] - 1, 0],
            [base_image.shape[0] - 1, base_image.shape[1] - 1],
        ]
    ).reshape(-1, 1, 2)

    merged_points = np.zeros((number_of_images, 4, 1, 2))
    merged_points[0, ...] = edge_points

    h_list = []
    for i, image in enumerate(input_images[1:]):
        src_pts, base_pts = get_matches(src_img=image, dst_img=base_image)
        H = ransac(
            src_pts=src_pts, dst_pts=base_pts, ransacReprojThreshold=5, maxIters=3000
        )
        h_list.append(H)
        merged_points[i + 1, ...] = perspective_transform(points=edge_points, homography_matrix=H)

    # edge points from all images
    merged_points = merged_points.reshape(-1, 2)

    min_x, max_x = int(min(merged_points[:, 0]) - 1), int(max(merged_points[:, 0]) + 1)
    min_y, max_y = int(min(merged_points[:, 1]) - 1), int(max(merged_points[:, 1]) + 1)

    aligned_base = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=int)
    aligned_base[
        -min_y : -min_y + base_image.shape[0], -min_x : -min_x + base_image.shape[1]
    ] = base_image
    # shift matrix
    H_translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    aligned_images = [aligned_base]
    for i, image in enumerate(input_images[1:]):
        aligned_image = apply_homography(
            image, H_translation @ h_list[i], (max_y - min_y, max_x - min_x)
        )
        aligned_images.append(aligned_image)
    # list of aligned images
    return aligned_images
