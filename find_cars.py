import numpy as np
import cv2
from features import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
from sklearn.svm import LinearSVC
import time
from sklearn.externals import joblib
from tracking import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

#dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
#svc = dist_pickle["svc"]
orient = 18
pix_per_cell = 8
cell_per_block = 2
spatial_size = (16,16)
hist_bins = 48
hist_range = (0,256)
hog_channel = "ALL"

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


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    #img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='BGR2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    bboxes = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.concatenate((spatial_features, hist_features, hog_features)).reshape(1,-1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append([(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)])
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img, bboxes

def train():
    v_names, nv_names = load_training_images_names()
    features_cars = extract_features(v_names,cspace='BGR2YCrCb',spatial_size=spatial_size,hist_bins=hist_bins,hist_range=hist_range,
                                        orient=orient,pix_per_cell=pix_per_cell,cell_per_block=cell_per_block, hog_channel=hog_channel)
    features_noncars = extract_features(nv_names,cspace='BGR2YCrCb',spatial_size=spatial_size,hist_bins=hist_bins,hist_range=hist_range,
                                        orient=orient,pix_per_cell=pix_per_cell,cell_per_block=cell_per_block, hog_channel=hog_channel)
    
    print(len(features_cars))
    X = np.vstack((features_cars,features_noncars)).astype(np.float64)
    print(len(X))
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    print(np.mean(scaled_X))

    # Define the labels vector
    y = np.hstack((np.ones(len(features_cars)), np.zeros(len(features_noncars))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    joblib.dump({"svc":svc,"scaler":X_scaler}, 'svc.pkl') 
    #joblib.dump(X_scaler, 'xScalar.pkl') 

    print(round(t2-t, 2), 'Seconds to train SVC...')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    
ystart = 370
ystop = 656
scale = 1
dict_svc= joblib.load("svc.pkl")
svc = dict_svc["svc"]
X_scaler = dict_svc["scaler"]
tracker = Tracker()

def process_image(img):
    draw_img = np.copy(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    out_img1, bboxes1 = find_cars(img, ystart, ystop, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    out_img2, bboxes2 = find_cars(img, ystart, ystop, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    out_img3, bboxes3 = find_cars(img, ystart, ystop, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    out_img4, bboxes4 = find_cars(img, ystart, ystop, 1.2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    bbox_all = []
    for x in bboxes1:
        bbox_all.append(x)
    for x in bboxes2:
        bbox_all.append(x)
    for x in bboxes3:
        bbox_all.append(x)
    for x in bboxes4:
        bbox_all.append(x)
    
    #bbox_all = np.concatenate((np.array(bboxes1),np.array(bboxes2),np.array(bboxes3)))
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = tracker.add_heat(heat,bbox_all)

    # Apply threshold to help remove false positives
    heat = tracker.apply_threshold(heat,2)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img, bboxes = tracker.draw_labeled_bboxes(draw_img, labels)

    detections = tracker.get_bbox_centers(bboxes)

    tracker.predict(detections,bboxes)
   
    for p in detections:
        cv2.circle(draw_img,p, 15, (0,0,255), -1)

    cv2.imshow("predict",draw_img)
    cv2.waitKey(0)
    #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    return draw_img


def test_images():
    files = glob.glob('test_images/*.jpg')
    for x in files:
        img = cv2.imread(x)

if __name__ == "__main__":
    #train()

    output_video = 'project_video.mp4'
    clip1 = VideoFileClip("test_video.mp4")
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(output_video, audio=False)



  