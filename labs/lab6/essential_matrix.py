import argparse

import PIL.ExifTags
import PIL.Image
from fundamental_matrix import *
from numpy import trace
from numpy.linalg import det
from numpy.linalg import matrix_rank as rank
from numpy import ndarray


def get_focal_length(img_path: str) -> float:
    """
    Parse metadata information from image.

    Args:
        img_path: path to image.

    Returns:
        focal length
    """
    img = PIL.Image.open(img_path)

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in img._getexif().items()
        if k in PIL.ExifTags.TAGS
    }

    return float(exif["FocalLength"])


def get_camera_intrinsics_matrix(
    focal_length: float, pixel_size: float, height: int, width: int
) -> ndarray:
    """
    Create camera intrinsic matrix.

    Args:
        focal_length: focal length of the camera
        pixel_size: size of the pixel in microns
        height: height of the image
        width: width of the image

    Returns:
        Camera intrinsics matrix. Shape (3,3).
    """
    f = (focal_length / pixel_size) * (10**3)
    intrinsic_matrix = np.array([[f, 0, (width / 2)], [0, f, (height / 2)], [0, 0, 1]])
    return intrinsic_matrix


def check_essential_matrix_constraints(E: ndarray):
    """
    For ideal essential matrix:
    det(E) = 0 and 2E(E.T)E - trace(E(E.T))E = 0.

    Args:
        E: Essential matrix. Shape (3,3).
    """
    print("0 = det(E) = ", det(E))
    print(
        "0 = 2 * E @ E.T @ E - trace(E @ E.T ) * E = \n",
        2 * E @ E.T @ E - trace(E @ E.T) * E,
    )


def find_essential_matrix(fundamental_matrix: ndarray, intrinsic_matrix: ndarray):
    """
    Calculate essential matrix.
    """
    print("Finding Essential matrix.")
    print("Input")
    print("     Fundamental matrix:  \n", fundamental_matrix)
    print("     Intrinsic matrix:  \n", intrinsic_matrix)

    print("  \n")

    print("Step 1. Calculating essential matrix.")
    essential_matrix = intrinsic_matrix.T @ fundamental_matrix @ intrinsic_matrix
    print(
        "essential_matrix = intrinsic_matrix.T @ fundamental_matrix @ intrinsic_matrix =  \n",
        essential_matrix,
    )

    print("  \n")

    print("Step 2. Make singular value decomposition of the essential_matrix.")
    print("U, S, Vh = np.linalg.svd(essential_matrix)")
    U, S, Vh = np.linalg.svd(essential_matrix)
    print("U  \n", U)
    print("S  \n", S)
    print("Vh  \n", Vh)

    print("  \n")

    print("Step 3. Updating sigma to have sigma1=sigma2>0 and sigma3=0")
    print("S1 = (S[0] + S[1]) * 0.5 * np.identity(3); S1[-1, -1] = 0")
    S1 = (S[0] + S[1]) * 0.5 * np.identity(3)
    S1[-1, -1] = 0
    print("np.diagonal(S1):   \n", np.diagonal(S1))

    print("  \n")

    print("Step 4. Updating essential matrix")
    updated_essential_matrix = U @ S1 @ Vh.T
    print("updated_essential_matrix = U @ S1 @ Vh.T =  \n", updated_essential_matrix)

    print("  \n")

    print(
        "Check rank of essential matrix: rank(updated_essential_matrix) = ",
        rank(updated_essential_matrix),
    )
    print("Check essential matrix constraint: ")
    check_essential_matrix_constraints(updated_essential_matrix)

    print("  \n")

    print("Calculating translation vector")
    translation_vector = ((S1[0, 0] + S1[1, 1]) * 0.5) * det(U) * Vh.T[2, :]
    print(
        "translation_vector = ((S1[0, 0] + S1[1, 1]) * 0.5) * det(U) * Vh.T[2, :]   \n",
        translation_vector,
    )

    print("  \n")

    print("Calculating rotation matrix")
    W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    print("W", W)
    rotation_matrix = det(U) * U @ W @ Vh.T * det(Vh.T)
    print("rotation_matrix = det(U) * U @ W @ Vh.T * det(Vh.T) =  \n", rotation_matrix)


def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_image", required=True, type=str, help="Left image")
    parser.add_argument("--right_image", required=True, type=str, help="Right image")
    parser.add_argument(
        "--epsilon", required=True, type=int, help="Threshold parameter in PIXELS"
    )
    parser.add_argument(
        "--n_iter",
        required=True,
        type=int,
        help="Number of iterations for RANSAC to find fundamental matrix.",
    )
    parser.add_argument(
        "--pixel_size", required=True, type=float, help="Size of pixel in microns."
    )

    args = parser.parse_args()

    img_left = cv2.imread(args.left_image, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(args.right_image, cv2.IMREAD_GRAYSCALE)

    assert img_left.shape[:2] == img_right.shape[:2], "Images dimensions do not match."

    left_pts, right_pts = get_matches(img_left=img_left, img_right=img_right)
    src_pts_h, dst_pts_h = convert_to_homogeneous(left_pts, right_pts)

    F_best, best_inliers, best_accuracy = get_best_fundamental_matrix(
        left_pts=src_pts_h,
        right_pts=dst_pts_h,
        epsilon=args.epsilon,
        n_iter=args.n_iter,
    )

    left_img_with_epiliens, right_img_with_epiliens = draw_epipolar_lines(
        left_pts=left_pts.copy(),
        right_pts=right_pts.copy(),
        img_left=img_left,
        img_right=img_right,
        F=F_best,
        inliers=best_inliers,
        epsilon=args.epsilon,
    )

    # save epipolar lines
    cv2.imwrite("left_img_with_right_epipolar_liens.png", left_img_with_epiliens)
    cv2.imwrite("right_img_with_left_epipolar_liens.png", right_img_with_epiliens)

    focal_length = get_focal_length(img_path=args.left_image)
    intrinsic_matrix = get_camera_intrinsics_matrix(
        focal_length=focal_length,
        pixel_size=args.pixel_size,
        height=img_left.shape[0],
        width=img_left.shape[0],
    )
    find_essential_matrix(fundamental_matrix=F_best, intrinsic_matrix=intrinsic_matrix)


if __name__ == "__main__":
    main()
