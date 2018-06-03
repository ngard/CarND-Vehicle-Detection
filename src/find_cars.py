import os, cv2, time, pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from datetime import datetime as dt

def read_movie(movie_path):
    cap = cv2.VideoCapture(movie_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret != True:
            break
        yield frame
    cap.release()

def mkdir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

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
def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
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
        
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'BGR2HLS':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if conv == 'BGR2LUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    index_image = 0

    img_original = img.copy()
    
    img = img.astype(np.float32)/255
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='BGR2YCrCb')
    ctrans_tosearch_RGB = convert_color(img_tosearch, conv='BGR2RGB')
    #ctrans_tosearch_LUV = convert_color(img_tosearch, conv='BGR2LUV')
    ctrans_tosearch_HLS = convert_color(img_tosearch, conv='BGR2HLS')
    #ctrans_tosearch = convert_color(img_tosearch, conv='BGR2HSV')
    #ctrans_tosearch = convert_color(img_tosearch, conv='BGR2LUV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        ctrans_tosearch_RGB = cv2.resize(ctrans_tosearch_RGB, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        ctrans_tosearch_HLS = cv2.resize(ctrans_tosearch_HLS, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    ch1_RGB = ctrans_tosearch_RGB[:,:,0]
    ch2_RGB = ctrans_tosearch_RGB[:,:,1]
    ch3_RGB = ctrans_tosearch_RGB[:,:,2]

    # ch1_LUV = ctrans_tosearch_LUV[:,:,0]
    # ch2_LUV = ctrans_tosearch_LUV[:,:,1]
    # ch3_LUV = ctrans_tosearch_LUV[:,:,2]

    
    ch1_HLS = ctrans_tosearch_HLS[:,:,0]
    ch2_HLS = ctrans_tosearch_HLS[:,:,1]
    ch3_HLS = ctrans_tosearch_HLS[:,:,2]

        
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    hog1_RGB = get_hog_features(ch1_RGB, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2_RGB = get_hog_features(ch2_RGB, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3_RGB = get_hog_features(ch3_RGB, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # hog1_LUV = get_hog_features(ch1_LUV, orient, pix_per_cell, cell_per_block, feature_vec=False)
    # hog2_LUV = get_hog_features(ch2_LUV, orient, pix_per_cell, cell_per_block, feature_vec=False)
    # hog3_LUV = get_hog_features(ch3_LUV, orient, pix_per_cell, cell_per_block, feature_vec=False)

    hog1_HLS = get_hog_features(ch1_HLS, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2_HLS = get_hog_features(ch2_HLS, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3_HLS = get_hog_features(ch3_HLS, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    rectangles = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

            hog_feat1_RGB = hog1_RGB[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2_RGB = hog2_RGB[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3_RGB = hog3_RGB[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

            # hog_feat1_LUV = hog1_LUV[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            # hog_feat2_LUV = hog2_LUV[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            # hog_feat3_LUV = hog3_LUV[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            
            hog_feat1_HLS = hog1_HLS[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2_HLS = hog2_HLS[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3_HLS = hog3_HLS[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3,
                                      hog_feat1_RGB, hog_feat2_RGB, hog_feat3_RGB,
                                      #hog_feat1_LUV, hog_feat2_LUV, hog_feat3_LUV,
                                      hog_feat1_HLS, hog_feat2_HLS, hog_feat3_HLS))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append((xbox_left,ytop_draw,win_draw,scale,ystart))
                # img_save = img_original[ytop_draw+ystart:ytop_draw+ystart+win_draw,xbox_left:xbox_left+win_draw]
                # img_save = cv2.resize(img_save,(window,window))
                # tstr = dt.now().strftime('%Y%m%d%H%M%S')
                # cv2.imwrite("../images/temp/%s-%06d.png"%(tstr,index_image),img_save)
                # index_image += 1
    return rectangles

def draw_boxes(img, rectangles):
    for xbox_left,ytop_draw,win_draw,scale,ystart in rectangles:
        cv2.rectangle(img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(255,0,),2)

def add_heat(heatmap, rectangles):
    # Iterate through list of bboxes
    for xbox_left,ytop_draw,win_draw,scale,ystart in rectangles:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        y_top = ytop_draw + ystart
        y_bottom = ytop_draw + ystart + win_draw
        x_left = xbox_left
        x_right = xbox_left + win_draw
        heatmap[y_top:y_bottom, x_left:x_right,2] += 1
        
    # Return updated heatmap
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

dist_pickle = pickle.load(open("/home/tohge/data/udacity.carnd/vehicle_detection/out/svc.pickle","rb"))

clf = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

is_write_out_movie = True
is_write_out_image = False

if is_write_out_movie:
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('detect_vehicle.avi',fourcc, 20.0, (1280,720))

num_frame = 0
    
for img in read_movie("../project_video.mp4"):
    rectangles = []
    heatmap = np.zeros_like(img).astype(np.float64)

    ystart = 400
    ystop = 480
    scale = 0.75
    rectangles.extend(find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    
    ystart = 400
    ystop = 500
    scale = 1.
    rectangles.extend(find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

    ystart = 400
    ystop = 550
    scale = 1.5
    rectangles.extend(find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

    ystart = 400
    ystop = 600
    scale = 2.0
    rectangles.extend(find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

    ystart = 400
    ystop = 650
    scale = 2.5
    rectangles.extend(find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    
    draw_boxes(img,rectangles)
    add_heat(heatmap,rectangles)
    cv2.threshold(heatmap,5,255,cv2.THRESH_TOZERO,heatmap)
    labels = label(heatmap)
    draw_labeled_bboxes(img,labels)

    if is_write_out_movie:
        video_writer.write(img)
    if is_write_out_image:
        cv2.imwrite("%5d.png"%num_frame,img)
        num_frame += 1
    
    cv2.imshow("img",img)
    cv2.imshow("heat",heatmap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if is_write_out_movie:
    video_writer.release()
