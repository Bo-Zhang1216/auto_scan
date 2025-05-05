# Data Aquisition
## recommended: create virtual environment python >= 3.10
### 1. Create a new conda environment named "auto_scan" with Python 3.10
conda create --name auto_scan python=3.10 -y

### 2. Activate it
conda activate auto_scan

### (Optional) 3. Verify you’re on the right Python version
python --version

## First install sam2 by the following steps:
!git clone https://github.com/facebookresearch/sam2.git && cd sam2
!pip install -e .
after this, rename the sam2 folder to sam2_repo and open the sam2_repo folder and move the sam2 folder out into the auto_scan folder

## getting datapoints

### folder
The folder of training data needs to be all images in the formats: "*.jpg","*.jpeg","*.png","*.bmp","*.tiff"

### valid flakes
Input the folder of images you want to use in the folder variable in "valid_falke_data.py" and run the program. The program will load all images in the folder and on each image, you will left click on the flakes you think is valid, the program will generate a point at where you clicked. If you think the point is good, press "s" to save that data point, you can save as many data points as you want per image and it is recommended to select multiple points on the flake at different locations such as edges and center. You can clear the point you put down by pressing "c", and you can navigate between the images with "a" and "d". You can also zoom in and out with + and -. When you are zoomed in, you can pan left/right/up/down with the arrow keys. At the end, press "ESC" to quit. The program may have delays for generating a point at where you clicked if the image is large(high resolution).

### invalid flakes and background using segmentation
**Important** make sure to run invalid_area_data.py in the auto_scan/sam2 directory

The "invalid_area_data.py" is used for getting the data points for the false values which are the flakes that are too thick and the background. Input the folder of trainning images and run the program. To generate data, you will left click to add a point for generating a segmentation. Please click on all the flakes that you think is valid. If you think there is at least a point on every valid flake, you can then press the space bar to generate a mask around the flakes you've selected(give some time for the mask generation). Make sure that all parts of the valid flakes are covered by the segmentation. It's okay if unvalid flakes and backgrounds are included by the segmentation but it is important that all parts of the valid flake is included in the mask (it's okay if only a tiny tiny part isn't, but it is prefered that all parts are included). If the mask isn't good, you can press "c" to clear all clicks and masks and restart. If you think the mask is good, press "s" to save the datapoints. You can navigate between images with a and d, and quit the program with "ESC"

## data labeling

Run the data_labeling.py file and you will get a final_data.json which you can use to train the model.

## Training

run the model.py file with the final_data.json file to get a model.

## Testing

to test the model on an image, in grid_test.py input the model path and image path and then run the program.