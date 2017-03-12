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
import copy


class Detector():
    def __init__(self):
        self.orient = 18
        self.pix_per_cel = 8
        self.cell_per_block = 2
        self.spatial_size = (32,32)
        self.hist_bins = 48
        self.hist_range = (0,256)
        self.hog_channel = "ALL"
        self.ystart = 385
        self.ystop = 656
        dict_svc= joblib.load("svc.pkl")
        self.svc = dict_svc["svc"]
        self.X_scaler = dict_svc["scaler"]
        self.count = 0
        self.no_cars_count = 0
        self.tracking = False
    
        # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self,img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

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

    def train(self, v_names, nv_names):
        #v_names, nv_names = load_training_images_names()
        features_cars = extract_features(v_names,cspace='BGR2YCrCb',spatial_size=self.spatial_size,hist_bins=self.hist_bins,hist_range=self.hist_range,
                                            orient=self.orient,pix_per_cell=self.pix_per_cel,cell_per_block=self.cell_per_block, hog_channel=self.hog_channel)
        features_noncars = extract_features(nv_names,cspace='BGR2YCrCb',spatial_size=self.spatial_size,hist_bins=self.hist_bins,hist_range=self.hist_range,
                                            orient=self.orient,pix_per_cell=self.pix_per_cel,cell_per_block=self.cell_per_block, hog_channel=self.hog_channel)

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

        self.svc = svc
        self.X_scaler = X_scaler

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


    def multi_scale_detection(self,img):
        out_img1, bboxes1 = self.find_cars(img, self.ystart, self.ystop, 1, self.svc, self.X_scaler, self.orient, self.pix_per_cel, self.cell_per_block, self.spatial_size, self.hist_bins)
        out_img2, bboxes2 = self.find_cars(img, self.ystart, self.ystop, 1.5, self.svc, self.X_scaler, self.orient, self.pix_per_cel, self.cell_per_block, self.spatial_size, self.hist_bins)
        out_img3, bboxes3 = self.find_cars(img, self.ystart, self.ystop, 2, self.svc, self.X_scaler, self.orient, self.pix_per_cel, self.cell_per_block, self.spatial_size, self.hist_bins)
        #out_img4, bboxes4 = self.find_cars(img, self.ystart, self.ystop, 1.2, self.svc, self.X_scaler, self.orient, self.pix_per_cel, self.cell_per_block, self.spatial_size, self.hist_bins)

        bbox_all = []
        for x in bboxes1:
            bbox_all.append(x)
        for x in bboxes2:
            bbox_all.append(x)
        for x in bboxes3:
            bbox_all.append(x)
        #for x in bboxes4:
        #    bbox_all.append(x)

        #bbox_all = np.concatenate((np.array(bboxes1),np.array(bboxes2),np.array(bboxes3)))
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = self.add_heat(heat,bbox_all)

        # Apply threshold to help remove false positives
        heat = self.apply_threshold(heat,2)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = copy.copy(img)
        bboxes = self.draw_labeled_bboxes(draw_img, labels)
        return bboxes

    def add_heat(self,heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
    
    def apply_threshold(self,heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap
    
    def draw_labeled_bboxes(self,img, labels):
        # Iterate through all detected cars
        bboxes = []
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            bboxes.append(bbox)
        # Return the image
        return bboxes

    def car_classify(self,test_img,bbox):

        x1 = bbox[0][0]
        x2 = bbox[1][0]
        y1 = bbox[0][1]
        y2 = bbox[1][1]

        cropped_img = test_img[y1:y2,x1:x2,:]

        feature = extract_feature(cropped_img,resize=(64,64),cspace='BGR2YCrCb',spatial_size=self.spatial_size,hist_bins=self.hist_bins,hist_range=self.hist_range,
                                            orient=self.orient,pix_per_cell=self.pix_per_cel,cell_per_block=self.cell_per_block, hog_channel=self.hog_channel)

        test_feature = self.X_scaler.transform(feature.reshape(1,-1))
        test_prediction = self.svc.predict(test_feature)
        return test_prediction