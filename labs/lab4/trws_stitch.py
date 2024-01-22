import argparse
import os
import time

import cv2
from aligning import align_images
from trws import create_panorama


def main():
    """Run stitching with TRWS."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-list",
        nargs="+",
        required=True,
        help="Image paths. First image is the Source image.",
    )
    parser.add_argument(
        "--n-iter-trws",
        type=int,
        required=True,
        help="Max iteration for TRWS algorithm",
    )
    parser.add_argument(
        "--out-img", type=str, required=True, help="Path to stitched image."
    )
    args = parser.parse_args()

    input_images = []
    for path in args.images_list:
        if not os.path.exists(path):
            raise FileExistsError(f"Image {path} doesn't exist.")
        input_images.append(cv2.imread(path))

    start_time = time.time()

    aligned_images = align_images(input_images)
    panorama = create_panorama(aligned_images, args.n_iter_trws)

    end_time = time.time()
    print("Total time: ", end_time - start_time)

    # save output image
    cv2.imwrite(args.out_img, panorama)


if __name__ == "__main__":
    main()
