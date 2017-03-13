# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[image1]: ./examples/scale1.5.jpg
[image2]: ./examples/scale2.jpg
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---
##Writeup / README:

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

- In the Detector class in detector.py the function `train` calls the `extract_features` function from features.py
- The `extract_features` function uses the skimage function `hog` to extract hog feature vectors from all the images at the file path passed in to the imgs parameter

####2. Explain how you settled on your final choice of HOG parameters.

- After testing I settled on using all the channels from the image in YCrCb colorspace by concatenating the hog features extracted from each channel
- Also after testing I found that using 18 orientations, 8 pixels per cell and 2 cells per block gave me the best results 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

- In the Detector class in detector.py the function `train` is used for training a linear SVM
- The features used are a concatenation of the hog features described above, with the color histogram (48 bins) and the spatial color features (32X32)
- Features were normalized using the sklearn `StandardScalar` class
- 20% of the training images were used for validation and were split using the sklearn `train_test_split` function

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

- I used the code provided in the lectures for the sliding window search. The `find_cars` function in the Detector class in detector.py shows this implementation. 
- For overlap I went with 75% overlap (2 cells per step), as this gave me the the best results. Anything lower did not give very reliable results.
- For scales I ended up using two (1.5,2) in the `multi_scale_detection` function in detector.py. This combination recognizes all the car close to the camera, and will sometimes pick up cars far ahead. 

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

- I used the YCrCb colorspace and used a combination of HOG, color histogram and color spatial information for features. I also normalized the features. Below are some example detections
![alt text][image1]
![alt text][image2]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./submission_video.mp4)
Also see this, [other video](./submission_video2.mp4), to see it working on a longer video


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

- I used the heatmap approach taught in the lectures. Where I added plus one to every pixel covered in a bounding box, as a result pixels with overlapping bounding boxes has values > 1. With this heatmap I thresholded with a value of 2 and then used the scipy function 'label' to get the new bounding boxes. 
- After combining overlapping bounding boxes using the heat map, I also run the classifier again on the new bounding box to verify that we actually got a car still.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- To start I followed the pipeline laid out in the lectures using a combination of Hog, color histogram and color spatial info for features to train a linear SVM. Then I used the heatmap method to deal with false positives.
- In addtion to this I added tracked object association between frames using the scipy function `linear_sum_assignment` which uses the Hungarian algorithm to get the min cost when assigning "workers" to "jobs". In this case I used the distance between bounding boxes as the cost. This allowed me to keep track of the position of cars between frames and calculate a average bounding box over frames. See the `associate` function in tracking.py for the implementation
- With data association between frames I decided to add Kalman filter state estimation to smooth the bounding box motion between frames as well as for tracking so I wouldn't have to run inefficient sliding window detection as often. For the kalman filter I am using a state of [x,y,x_velocity,y_velocity]. The kalman filter implementation is in the `TrackedObject` class in tracking.py. 
- For how I integrated the Kalman filter into my main program see the `process` function in find_cars.py. My program follows this flow:
    - If Detect 
      - Run Kalman filter prediction step for all tracked objects
      - Do data association between detected frames and all tracked frames
      - Run the Kalman filter measurement step for all tracked objects still in frame
    - If Track
      - Run Kalman filter prediction step for all tracked objects
- My current implementation does not run in real time, nor does it take into account when tracked objects become occluded. Using the already implemented Kalman filter I could track objects through occlusions, but did not have time to implement it for this project. Using the camshift tracking algorithm would also help reduce the number of times I have to run the detection method and speed things up. But the sliding window detection step is still a major hurtle for running in real time. 

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.


