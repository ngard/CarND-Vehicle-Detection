**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/windows.png
[image21]: ./output_images/test1.jpg
[image22]: ./output_images/test2.jpg
[image23]: ./output_images/test3.jpg
[image24]: ./output_images/test4.jpg
[image25]: ./output_images/test5.jpg
[image26]: ./output_images/test6.jpg
[image31]: ./output_images/heatmap_test1.jpg
[image32]: ./output_images/heatmap_test2.jpg
[image33]: ./output_images/heatmap_test3.jpg
[image34]: ./output_images/heatmap_test4.jpg
[image35]: ./output_images/heatmap_test5.jpg
[image36]: ./output_images/heatmap_test6.jpg
[image4]: ./output_images/falsepositive.png
[video1]: ./detect_vehicle.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained or in `train_svc()` in `src/train_svc.py:48`.

This function reads provided vehicle and non-vehicle images, extracts features of each images and finally trains SVM with the images.

Feature extraction is executed in `extract_features()` in `src/train_svc.py:12`.

In this function, each images are converted from BGR to YCrCb color space with `convert_color()` in `src/common.py:9`
and extracted binned spatial features (`bin_spatial()` in `src/common.py:24`),
color histogram features (`color_hist()` in `src/common.py:31`) and hog features (`get_hog_featues()` in `src/common.py:41`).

####2. Explain how you settled on your final choice of HOG parameters.

At first, I decided to use all of provided features (binned spatial, color histogram and HOG features.)

For color space, I tried several color spaces (i.e. RGB, YUV and HLS) and achieved best performance with YCrCb color space.

Then I noticed it performs well (more than 99% accuracy on test sets) even on the default HOG parameter (`orientation=8`, `pixels_per_cell=(8,8)` and `cells_per_block=(2,2)`),
so I decided to use that.

I also tried to use different color space aside with YCrCb to expect better accuracy but it takes so much time to training and prediction but resulted in worse result (probably overfitting).

After starting to predict on the project movie, I suffered from slow computation. I profiled the code and found that calculating color histogram features are the bottleneck.
Then I tried without color histogram features and found that the result does not change so much.

Finally, I ended up with a combination of binned spatial features and HOG features in YCrCb color space.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Training is executed in the latter half of `train_svc()` in `src/train_svc.py`.

After extracting features from dataset, features are normalized with `sklearn.preprocessor.StandardScaler`.

Then all the elements are shuffled randomly and devided into training set and test set with `sklearn.model_selection.train_test_split()`.

Finally, LinearSVC is trained with the training set and trained SVM is saved in a pickle file.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I prepared 5 sizes of sliding windows (scale: 1.0, 1.25, 1.5, 2.0 and 2.5) and they are configured as the image below to meet the purpose and reduce processing time.

Larger windows are configured to detect near vehicles (usually in the lower half of the image) and smaller windows are configured to detect far vehicles (usually in the middle of the image).

Also, I configured them densely by overlapping them frequently so as to maximize the number of boxes to detect one vehicle for the reason mentioned in the next section.

![All the windows][image1]
---

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

These are the example output and heatmap of test images. Blue boxes are regions where SVC predicted that a vehicle is in the region.

As the accuracy of SVC is high enough even without any tweak, the number of false positives are not so many and does not appear in the same region so much.
On the other hand, lots of true positive blue boxes appear around vehicles.

Therefore, I decided to filter out false positives by maximizing the threashold of heatmap, the number of detected boxes in a region.
Heatmaps are calcurated in `add_heat()` in `find_cars.py:166`. Then the region with low heat is filtered out by `cv2.threashold()` in `find_cars.py:268`.
The threashold is empirically determined to 4.

Then the high heat regions of a heatmap are grouped and labeled by `scipy.ndimage.measurements.label()` in `find_cars.py:269`.
Finally, new bound boxes of each regions are calculated by `extract_labeled_bboxes()` in `find_cars.py:180` and the results are showed as orange boxes.

![detection result of test1.jpg][image21]
![heatmap result of test1.jpg][image31]
![detection result of test2.jpg][image22]
![heatmap result of test2.jpg][image32]
![detection result of test3.jpg][image23]
![heatmap result of test3.jpg][image33]
![detection result of test4.jpg][image24]
![heatmap result of test4.jpg][image34]
![detection result of test5.jpg][image25]
![heatmap result of test5.jpg][image35]
![detection result of test6.jpg][image26]
![heatmap result of test6.jpg][image36]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result][video1].

The red boxes are the final result of my vehicle detection logic.

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Even though most of false positives are filtered out with intra-frame heatmap technique explained in the previous section,
there are still a few remaining false positives as exemplified below.

![An example of false positive even after heatmap filter][image4]

To filter them out, I decided to introduce inter-frame tracking technique and implemented as `tracking_detection()` in `find_cars.py:209`.

In the function, the intra-frame results of each frame are compared to that of previous frame by checking the center of bbox is in the bbox of previous frame.
If such result is found, it keeps the same ID to the previous frame and increment the tracking count by 1 from that of previous frame.

Then by filtering out the results of low count (by changing the bbox color at `draw_bbox()` in `src/find_cars.py:196`),
the logic filters out all the false positives in the video. The threashold of this filter was empirically set to 6.

By applying these techniques, my logic keeps track of only the vehicles in most part of the project movie.

However, the classifier misses to detect vehicles very rarely.

For that case, I also implemented lost-tracking mode in the latter half of `tracking_detection` in `src/find_cars.py:241`.
With this technique, even once the classfier missed to detect a vehicle, the tracker keeps tracking for a few frames. The duration of lost tracking is depending on the tracking count of the vehicle, the more the count is the longer the tracker keeps to track.
This is enabled by divide the tracking counter by 10 in lost tracking mode.

After all, my logic keeps track of only the vehicles, successfully.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The thing tormented me the most is computation time.
As I implemented this project with Python, it takes so much time in each line and it is so hard to scale in many-core machines because of GIL.
Even though I tried so much to use other cores in my machine by multiprocessing or joblib, I ended up without getting faster computation.
Therefore, I had to minimize the length of feature vector and the number of region to apply SVC.
Luckily, that was enough to solve this project, however, longer feature vectors and more regions are preferred to be used to get good result in more variety of scenes.
To make it possible within reasonable time, more training data and faster computation is required.
For more training data, I have already implemented `src/crop_images.py` to cut out more images for training from CrowdAI & Udacity dataset which I did not need to use.
For boosting the calcuration, using CuPy and/or C++ with multi-threading is my idea so far.

Another weak point of this logic is incapability of distinguishing several vehicles if they are in almost the same direction.
To deal with it, better segmentation techniques should be used.