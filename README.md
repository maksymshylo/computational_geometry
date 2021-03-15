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
## Lab 3 - Panorama view
### Homography estimation with RANSAC
#### Examples
```bash
python3 lab3/homography.py lab3/test_images lab3/test_images MAX_ITER  PATH_OUT_IMG

python3 lab3/homography.py lab3/test_images/book_1.png lab3/test_images/book_2.png 2000 lab3/test_images/out.png
```
## Lab 4 - Panorama view with Tree Reweighted Message Passing (TRW-S) as a stitching method
#### Examples
Important: the first image for input is the base image for applying homography
```bash
python3 lab4/panorama.py trws_n_iter images

python3 lab4/panorama.py 5 lab4/test_images/book2.jpg lab4/ttest_images/book1.jpg lab4/test_images/book3.jpg
python3 lab4/panorama.py 5 lab4/test_images/ball2.jpg lab4/test_images/ball3.jpg lab4/test_images/ball1.jpg
```
