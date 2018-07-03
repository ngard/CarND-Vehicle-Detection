import os, cv2, time, pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from datetime import datetime as dt
from common import mkdir_if_not_exists, convert_color, bin_spatial, color_hist, get_hog_features

def read_movie(movie_path, speed = 1):
    cap = cv2.VideoCapture(movie_path)
    count_frame = 0
    while(cap.isOpened()):        
        ret, frame = cap.read()
        if ret != True:
            break

        count_frame += 1
        if not count_frame % speed == 0:
            continue
        
        yield frame
    cap.release()

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    index_image = 0

    img_original = img.copy()
    
    img = img.astype(np.float32)/255
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='BGR2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    rectangles = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3
                ))

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
speed = 20

if is_write_out_movie:
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('detect_vehicle.avi',fourcc, 20.0, (1280,720))

count_frame = 0
    
for img in read_movie("./project_video.mp4",speed):
    count_frame += 1
    
    rectangles = []
    heatmap = np.zeros_like(img).astype(np.float64)

    # ystart = 400
    # ystop = 480
    # scale = 0.75
    # rectangles.extend(find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    
    # ystart = 400
    # ystop = 500
    # scale = 1.
    # rectangles.extend(find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

    ystart = 375
    ystop = 500
    scale = 1.3
    rectangles.extend(find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    
    ystart = 380
    ystop = 550
    scale = 1.5
    rectangles.extend(find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

    ystart = 385
    ystop = 600
    scale = 2.0
    rectangles.extend(find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

    ystart = 390
    ystop = 650
    scale = 2.5
    rectangles.extend(find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    
    draw_boxes(img,rectangles)
    add_heat(heatmap,rectangles)
    cv2.threshold(heatmap,3,255,cv2.THRESH_TOZERO,heatmap)
    labels = label(heatmap)
    draw_labeled_bboxes(img,labels)

    if is_write_out_movie:
        video_writer.write(img)
    if is_write_out_image:
        cv2.imwrite("%5d.png"%count_frame,img)
    
    cv2.imshow("img",img)
    cv2.imshow("heat",heatmap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if is_write_out_movie:
    video_writer.release()
