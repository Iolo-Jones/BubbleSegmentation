# BubbleSegmentation

BubbleSegmentation is software that identifies bubbles in images, using a scale-equivariant convolutional kernel. This results in a bubble heatmap for each radius, which is thresholded to pick out a 'constellation' of points at each bubble centre. The threshold level varies as a function of radius and bubble volume fraction, which was derived empirically from sample data.

## Repository guide

1. bubbles.py contains all the main functions
2. utils_visualisation.py has utility and visualisaiton functions
3. bubble_processing.ipynb is the guided notebook for using the software
4. bubble_test_notebook.ipynb is the working test notebook
5. bubble_data is the folder where bubble_processing.ipynb will save the computed bubble data
6. threshold_testing.py has the empirical measurements used to create the thresholding model

## How to use BubbleSegmentation

1. Clone the repository and create a new folder called 'data' inside the main one. Put all your images in subfolders in there. I.e. the paths of your images should look like 'data/05.17_0.37/230517_AR_1632.jpg'.
2. Install openCV if you haven't already. The other dependencies are standard.
3. Open bubble_processing.ipynb and follow the instructions, working through each experiment folder at a time.
