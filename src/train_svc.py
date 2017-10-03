import os, cv2, time, pickle
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features                        
        
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32), hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        feature_image_YCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        feature_image_RGB = image
        feature_image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        feature_image_LUV = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []

                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image_YCrCb[:,:,channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))

                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image_RGB[:,:,channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))

                # for channel in range(feature_image.shape[2]):
                #     hog_features.append(get_hog_features(feature_image_LUV[:,:,channel],
                #                                          orient, pix_per_cell, cell_per_block,
                #                                          vis=False, feature_vec=True))

                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image_HLS[:,:,channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)


            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def train_svc(images_vehicle,images_nonvehicle, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat):
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    
    t=time.time()
    features_YCrCb_vehicle = extract_features(images_vehicle, cspace="YCrCb",
                                              spatial_size=spatial_size, hist_bins=hist_bins,
                                              orient=orient, pix_per_cell=pix_per_cell,
                                              cell_per_block=cell_per_block,
                                              hog_channel=hog_channel, spatial_feat=spatial_feat,
                                              hist_feat=hist_feat, hog_feat=hog_feat)
    features_YCrCb_nonvehicle = extract_features(images_nonvehicle, cspace="YCrCb",
                                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                                 orient=orient, pix_per_cell=pix_per_cell,
                                                 cell_per_block=cell_per_block,
                                                 hog_channel=hog_channel,
                                                 spatial_feat=spatial_feat,
                                                 hist_feat=hist_feat, hog_feat=hog_feat)
    features_vehicle = features_YCrCb_vehicle
    features_nonvehicle = features_YCrCb_nonvehicle
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')

    X = np.vstack((features_vehicle, features_nonvehicle)).astype(np.float64)

    bad_indices = np.where(np.isinf(X))
    print("INF indices:",bad_indices)
    bad_indices = np.where(np.isnan(X))
    print("NaN indices:",bad_indices)
    
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(features_vehicle)), np.zeros(len(features_nonvehicle))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    return svc, X_scaler
                            
def list_images(category):
    image_paths = []
    dir_root = "../images/"
    for directory in os.listdir(dir_root+category):
        directory = dir_root + category + "/" + directory
        if not os.path.isdir(directory):
            continue
        for fname_image in os.listdir(directory):
            if not fname_image.endswith(".png"):
                continue
            image_paths.append(directory+"/"+fname_image)
    return image_paths
    
images_vehicle = list_images("vehicles")
images_nonvehicle = list_images("non-vehicles")

print(len(images_vehicle),len(images_nonvehicle))

orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

clf, X_scaler = train_svc(images_vehicle,images_nonvehicle,orient,pix_per_cell,cell_per_block,
                          spatial_size,hist_bins,spatial_feat,hist_feat,hog_feat)

dist_pickle = {"svc":clf,
               "scaler":X_scaler,
               "orient":orient,
               "pix_per_cell":pix_per_cell,
               "cell_per_block":cell_per_block,
               "spatial_size":spatial_size,
               "hist_bins":hist_bins}

pickle.dump(dist_pickle,open("svc.pickle","wb"))

