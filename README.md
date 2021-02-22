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

python3 lab3/homography.py lab3/test_images/im_1.png lab3/test_images/im_2.png 2000 lab3/test_images/out.png
```
