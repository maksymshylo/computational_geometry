# Computational Geometry for Computer Vision
Labs for a University Course     

## Lab 1 - Basic CV algorithms using opencv
### stereo on unaligned images
## Lab 2 - Stereovision using SGM algorithm
### stereovision
#### Examples
```bash
python3 lab2/sgm_stereo.py lab2/test_images lab2/test_images smoothing coefficient min_dx max_dx min_dy max_dy

python3 lab2/sgm_stereo.py lab2/test_images/im0.ppm lab2/test_images/im1.ppm 3 0 20 0 0
```
## Lab 3 - Panorama View
### Homography estimation with RANSAC
#### Examples
```bash
python3 lab3/homography.py lab3/test_images lab3/test_images MAX_ITER  PATH_OUT_IMG

python3 lab3/homography.py lab3/test_images/book_1.png lab3/test_images/book_2.png 2000 lab3/test_images/out.png
```
## Lab 4 - Panorama View With Tree Reweighted Message Passing (TRW-S) as a stitching method
#### Examples
Important: the first image for input is the base image for applying homography
```bash
python3 lab4/panorama.py trws_n_iter images

python3 lab4/panorama.py 5 lab4/test_images/book2.jpg lab4/ttest_images/book1.jpg lab4/test_images/book3.jpg
python3 lab4/panorama.py 5 lab4/test_images/ball2.jpg lab4/test_images/ball3.jpg lab4/test_images/ball1.jpg
```
## Lab 5 - Finding Fundamental Matrix With 7-point Algorithm (RANSAC)
#### Examples
```bash
python3 lab5/fundamental_matrix.py left_image right_image epsilon_in_pixels number_of_iterations

python3 lab5/fundamental_matrix.py lab5/test_images/bike1.jpeg lab5/test_images/bike2.jpeg 1 1000
python3 lab5/fundamental_matrix.py lab5/test_images/mount1.jpeg lab5/test_images/mount2.jpeg 1 1000
python3 lab5/fundamental_matrix.py lab5/test_images/messL.png lab5/test_images/messR.png 1 1000

```
## Lab 6 - Finding Essential Matrix with Fundamental Matrix and Camera Intrinsics Parameters
#### Examples
```bash
python3 lab6/essential_matrix.py left_image right_image epsilon_in_pixels number_of_iterations pixel_size_in_microns

python3 lab6/essential_matrix.py left_image right_image 1 1000 0.8

```
