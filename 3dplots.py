import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import glob 

def load_training_images_names():
    vehicles = []
    non_vehicles = []
    vehicles.append(glob.glob('training_imgs/vehicles/vehicles/GTI_Far/*.png'))
    vehicles.append(glob.glob('training_imgs/vehicles/vehicles/GTI_Left/*.png'))
    vehicles.append(glob.glob('training_imgs/vehicles/vehicles/GTI_MiddleClose/*.png'))
    vehicles.append(glob.glob('training_imgs/vehicles/vehicles/GTI_Right/*.png'))
    vehicles.append(glob.glob('training_imgs/vehicles/vehicles/GTI_Right/*.png'))
    vehicles.append(glob.glob('training_imgs/vehicles/vehicles/KITTI_extracted/*.png'))

    non_vehicles.append(glob.glob('training_imgs/non-vehicles/non-vehicles/Extras/*.png'))
    non_vehicles.append(glob.glob('training_imgs/non-vehicles/non-vehicles/GTI/*.png'))

    vehicles_out = np.concatenate(vehicles)
    non_vehicles_out = np.concatenate(non_vehicles)
    
    return vehicles_out, non_vehicles_out

def plot3d(pixels, colors_rgb, 
        axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]): 
    """Plot pixels in 3D.""" 
 
    # Create figure and 3D axes 
    fig = plt.figure(figsize=(8, 8)) 
    ax = Axes3D(fig) 
 
    # Set axis limits 
    ax.set_xlim(*axis_limits[0]) 
    ax.set_ylim(*axis_limits[1]) 
    ax.set_zlim(*axis_limits[2]) 
 
    # Set axis labels and sizes 
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8) 
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16) 
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16) 
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16) 
 
    # Plot pixel values with colors given in colors_rgb 
    ax.scatter( 
        pixels[:, :, 0].ravel(), 
        pixels[:, :, 1].ravel(), 
        pixels[:, :, 2].ravel(), 
        c=colors_rgb.reshape((-1, 3)), edgecolors='none') 
 
    return ax  # return Axes3D object for further manipulation 
 
 
# Read a color image 
 
v,nv = load_training_images_names()

for x in nv:
    img_small = cv2.imread(x) 


    # Select a small fraction of pixels to plot by subsampling it 
    #scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns 
    #img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST) 
 
    # Convert subsampled image to desired color space(s) 
    img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB 
    img_small_YCrCb = cv2.cvtColor(img_small, cv2.COLOR_BGR2YCrCb)
    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV) 
    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting 
 
    # Plot and show 
    plot3d(img_small_YCrCb, img_small_rgb, axis_labels=list("YCrBr")) 
    plt.show() 
 
    plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV")) 
    plt.show()