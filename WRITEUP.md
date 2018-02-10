# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HOG_car.png
[image2]: ./output_images/HOG_not_car.png
[image3]: ./output_images/car_found1.png
[image4]: ./output_images/car_found2.png
[image5]: ./output_images/car_found3.png
[image6]: ./output_images/car_found4.png
[image7]: ./output_images/car_found5.png
[image8]: ./output_images/bboxes_and_heat.png
[image9]: ./output_images/labels_map.png
[image10]: ./output_images/output_bboxes.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The HOG feature extractor is a way to extract meaningful features of a image. If we look through the different channels of the hog_feature map we can see that it captures the outlines of a car. It's like detecting a face in a picture. Hence the hog feature is extremely valuable to identify a car and also identify what's not a car.

HOG stands for “Histogram of Oriented Gradients”; it divides an image in several pieces as demonstrated in the image below. For each piece the HOG classifier calculates the gradient of variation for given orientations in those pieces. The gradient is the variation of colors just like a contrast filter. It’s a partial derivative and direction is controlled by the number of orientations.

I ran it through all the 3 feature extractors that we studied in the lectures. The HOG classifier followed by the spatial features and the histogram of the color channels. This gives us 3 separate set of feature arrays about the image and we concatenate these 3 arrays and return that.
 
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Car:
![alt text][image1]

Not Car:
![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I used what was provided in the lectures and worked based off of that. I realized early on that using 'RGB' was taking more time and giving less accuracy through my training model. I then tried others as well and turned out that 'YCrCb' was faster and had higher accuracy. I switched all my functions to use that and the HOG classifier as well.  

The less the number of pixels per cell gives out a better result but increases the time of training the model. I  was using orient 8 and that was good enough to recognize a car; so I increased it to 9 and the result was still good so I used that. 

I used the cell per block as it was in the lecture but I decreased my spatal size to (16, 16) and hist_bins to 16 from it's initial value (32,32) and 32 respectively. The model was taking atleast twice the time for high spatial size and (16,16) was giving me a 0.98 accuracy which is quite great whereas (32,32) dropped the accuracy to 0.92.

My final choice of params for HOG parameters.

```python
colorspace='YCrCb'
orient=9
pix_per_cell=8
cell_per_block=2
hog_channel='ALL'
spatial_size=(16, 16)
hist_bins=16
hist_range=(0, 256)
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used the same classifier just like in the lecture. I first converted the image to 'YCrCb' and then ran the feature extractor mentioned above with my params also mentioned above. The feature extractor is a combination of hog feature extractor, spatial color feature extractor and histogram of the color channels feature extractor respectively.

I read the entire data-set of images for vehicles and non-vehicles in two different arrays. I used a Support Vector Machine Classifier (SVC), with linear kernel, based on function SVM from scikit-learn powered by the feature-extractor logic containing the HOG detector. All that the SVC does it forms a common divide between two or more data-sets based on uncommon features between them and common features in an individual set. We can then randomly ask this classifier if an input is that classified object or not. In this case we created two different classifying one for vehicle and other for non-vehicle. We can get the classifier and the scaler from SVC and use that later on to identify cars in images.  

Wiki on SVC: https://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html

The classifier worked pretty well with the parameters mentioned above and I tweaked the parameters to get a better result which was an accuracy of 0.98 and in 12.0 to 14.0 secs.

I saved the classifier and scaler in file using the pickle library so I don't have to re-run my classifier again and again.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the code from the lecture implement a sliding window approach which classifies each window as either car or non-car.

I didn't use most of the top and some of the bottom part of the image as I don't need to detect cars in the sky and the partial parts of cars that appear right beside our car. I then resized the image 0.66% of it's original size that will make sliding windows a lesser effort.

We then start traversing the image with our window. The windows of 64 sampling rate with 8 cells and 8 pixels per cell and start moving our window by 2 which leads to 75% overlap. 

I wanted to move move my window with hardly any overlap but most of the cars afar and nearby had two boxes on them on both the minimum number of overlap was around 2 so I stuck to it.

We run the HOG classifier on every window and classify it as a car or not. If it is a car we put that window in a box list and draw a box for the window on the image.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

There's no real pipeline for this code I just use the Sliding window to classify a Window into car or not and store that window location draw a window on the image for it. I'll go through how I found the car on an image and then how I applied heatmap for videos.

1. I defined my features which was HOG features of the three HSL channels. I defined the parameters for
this hog features. 
2. I used the Feature Extractor defined above in HOG classifier to extract the features of a given image.
3. I ran features from step 2 through the linear SVC classifier with an X scaler that I had trained earlier with my selected features which will be used predict if the features of an image is a vehicle. 
4. Using the Sliding window discussed above on the image ​which will be used to carve out windows on the image where the classifier predicts that a vehicle is located or not. We slide this window within different parts of the image and store the boxes in which the car was found. 
5. To process the video I used the find_cars sliding window with two scales i.e. 1.0 and 1.5 and consolidated the images. For some frames the overlap ratio is not big enough so I increased of the entire find_cars function.
6. For video I got these boxes that are positive accumulated over a series of consecutive frames and feed it to a HeatMap. ​ Areas with multiple detections get “hot” while transient false positives stay “cool” as we impose a threshold to this HeatMap​. The remaining “Hot” regions and draw a bounding box for each “hot region” which implies a vehicle 

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I did a sanity check. By a sanity check I mean we know in a window that if a car is in one spot then in the proceeding frames that car won't jump to a different spot it'll gradually move in the same region. So in a video I built a heatmap to combine overlapping detections and remove false positives. To make a heat map we start with a blank grid and “add heat” (+1) for all pixels within windows where positive detections are reported by the classifier. The “hotter” the parts, the more likely it is a true detection, and we can then force anything below a threshold to be rejected as false positives. We
have integrated a heat map over several frames of video. Areas with multiple detections get “hot” while transient false positives stay “cool”. I have used a threshold or how many frames it will keep before rejecting the oldest frame. I used `scipy.ndimage.measurements.label()` ​to identify individual blobs in the heatmap, each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

Here are is the image frames and their corresponding heatmaps and applied labels after using a threshold of 1:

![alt text][image8]
![alt text][image9]
![alt text][image10]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the start I was getting my boxes all over the cars but even above and below the cars. Tweaking the parameters fixed that and that took a major amount of my time. The second issue was tweaking the heatmap. This solution was only achieved because I could see the training data and the inputs which made the tuning of the parameters very biased. I also can see the cars are very slow so the we are able to detect them. I am not sure how this will work in night time or even within a tunnel. Also we are only detecting cars and not bike or trucks. I think this can be much better than what it is right now.
