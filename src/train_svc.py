import os, cv2, time, pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from common import bin_spatial, color_hist, get_hog_features, convert_color

base_dir = "/home/tohge/data/udacity.carnd/vehicle_detection"

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='BGR2RGB', spatial_size=(32, 32), hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'

        feature_image = convert_color(image,cspace)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            hog_features = []

            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def train_svc(images_vehicle,images_nonvehicle, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat):
    t=time.time()
    features_vehicle = extract_features(images_vehicle, cspace="BGR2YCrCb",
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
    features_nonvehicle = extract_features(images_nonvehicle, cspace="BGR2YCrCb",
                                           spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block,
                                           spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)
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
    dir_root = base_dir + "/in/images/"
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
hist_feat = False # Histogram features on or off
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

pickle.dump(dist_pickle,open(base_dir+"/out/svc.pickle","wb"))
