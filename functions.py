# GLOBAL IMPORTS:
from typing import Union
from glob import glob
import os 
os.environ["OMP_NUM_THREADS"] = '1'

# CLASSIC LIBRARIES:
import numpy as np
import matplotlib.pyplot as plt

# LOAD IMAGES:
import PIL
from PIL import Image
# import display from IPython:
from IPython.display import display

# DISPLAY IMAGES/IMAGE PROCESSING:
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.filters import gabor
from scipy.stats import skew, kurtosis
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

# SKLEARN:
# import VarianceThreshold from sklearn:
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

# KMEANS:
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler



# GLOBAL VARIABLES:
# Include the path to the folder containing the images of the project description:
PATH_EXAMPLE = "data_project/train2/"

""" VISUALIZATION FUNCTIONS """

# Show the images in the notebook:
def show_image(image: Union[str, np.ndarray]):
    if isinstance(image, str):
        image = PIL.Image.open(image)
    # display images in smaller size:
    image.thumbnail((400, 400), PIL.Image.LANCZOS)
    display(image)

# Show the images from the PATH_EXAMPLE folder:
def show_example_images():
    for image_path in glob(PATH_EXAMPLE + "*.png"):
        # show only the first image example:
        show_image(image_path)
        break

""" LOADING and SAVING FUNCTIONS """

# Utils functions (Loading images and saving the solution puzzles):

def load_input_image(image_index,  folder ="train2", path = "data_project"):
    
    filename = "train_{}.png".format(str(image_index).zfill(2))
    path_solution = os.path.join(path,folder, filename )
    
    im= Image.open(os.path.join(path,folder,filename)).convert('RGB')
    im = np.array(im)
    return im

def save_solution_puzzles(image_index , solved_puzzles, outliers, folder ="train2", path = "data_project", group_id = 0):
    
    path_solution = os.path.join(path,folder + "_solution_{}".format(str(group_id).zfill(2)))
    if not  os.path.isdir(path_solution):
        os.mkdir(path_solution)

    print(path_solution)
    for i, puzzle in enumerate(solved_puzzles):
        filename =os.path.join(path_solution, "solution_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)))
        Image.fromarray(puzzle).save(filename)

    for i , outlier in enumerate(outliers):
        filename =os.path.join(path_solution, "outlier_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)))
        Image.fromarray(outlier).save(filename)

def solve_and_export_puzzles_image(image_index , folder = "train2" , path = "data_project"  , group_id = "00"):
    """
    Wrapper function to load image and save solution
            
    Parameters
    ----------
    image:
        index number of the dataset

    Returns
    """

      # open the image
    image_loaded = load_input_image(image_index , folder = folder , path = path)
    #print(image_loaded)
    
   
    ## call functions to solve image_loaded
    solved_puzzles = [ (np.random.rand(512,512,3)*255).astype(np.uint8)  for i in range(2) ]
    outlier_images = [ (np.random.rand(128,128,3)*255).astype(np.uint8) for i in range(3)]
    
    save_solution_puzzles (image_index , solved_puzzles , outlier_images , folder = folder ,group_id =group_id)

    
    return image_loaded , solved_puzzles , outlier_images


""" IMAGE PROCESSING FUNCTIONS """

# Segment the image using Canny edge detection:
def segment_image_canny(img):
    """
    Segment the image to retrieve the tiles.
    """
    # 1. Normalize the image:
    norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # We are finally not converting the image to grayscale:
    # train_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. Filter the image with a Gaussian filter:
    blur = cv2.GaussianBlur(norm, (5, 5), 0)

    # 2. Apply Canny edge detection:
    auto = cv2.Canny(blur, 35, 40)

    return auto

# Apply Morphologicals operators to fill the holes / get rid of the noise:
def apply_mat_morph(mask):
    """
    Apply morphological operations to the mask to remove noise and fill in holes
    """
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)) # dilate the mask to fill in holes
    #kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6)) # erode the mask to get rid of noise
    #kernel_close =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) # close the mask
    #kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) # open the mask

    mask_dilate = cv2.dilate(mask.astype(np.uint8), kernel_dilate, iterations=2)
    #mask_erode = cv2.erode(mask_dilate, kernel_erode, iterations=7)
    #mask_open = cv2.morphologyEx(mask_erode, cv2.MORPH_OPEN, kernel_open, iterations=6)
    #mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=6)

    # 1. Close the image:
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) # close the mask

    mask_close = cv2.morphologyEx(mask_dilate.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close, iterations=5)

    #2. Binary fill holes:
    mask_erode = binary_fill_holes(mask_close).astype(np.uint8)

    return mask_erode

# Get the contours of the objects in the mask:
def get_contours(img, mask):
    """
    Get the contours of the objects in the mask
    """
    #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = mask
    # Find the contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # If the contours area is bigger than 1000, keep it
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    final_contours = []
    coords = []
    im_draw = img.copy()
    tiles = []

    for cnt in contours:
        # cnt is a list of points of the contour of a tile (x,y) coordinates of the contour, ordered clockwise
        x1, y2 = cnt[0][0] # top left corner
        approx = cv2.approxPolyDP(cnt, 0.09*cv2.arcLength(cnt, True), True) # approximates the contour with accuracy proportional to the contour perimeter

        if len(approx) <= 5: # if the contour has at least 4 corners
            rect = cv2.minAreaRect(cnt) # get the rectangle that encloses the contour
            (x, y), (w, h), a = rect # get the coordinates of the center, width, height, and angle of rotation of the rectangle
            ratio = float(w)/h # compute the ratio of the width to the height of the rectangle
            if 0.8 < ratio < 1.2: # if the ratio is close to 1, the rectangle is a square, and the contour is a tile
                # the order of the box points: bottom left, top left, top right,
                # bottom right
                box = cv2.boxPoints(rect) # get the coordinates of the corners of the rectangle
                box = np.int0(box) # convert the coordinates to integers
                if cv2.contourArea(box) > 8000: # if the area of the rectangle is large enough, the contour is a tile
                    im_draw = cv2.drawContours(im_draw, [box], 0, (0, 255, 0), 4) # draw contours in green color
                     # get width and height of the detected rectangle
                    width = int(rect[1][0])
                    height = int(rect[1][1])
                    src_pts = box.astype("float32")
                    # coordinate of the points in box points after the rectangle has been
                    # straightened
                    dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

                    # the perspective transformation matrix
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                    # directly warp the rotated rectangle to get the straightened rectangle
                    warped = cv2.warpPerspective(img, M, (width, height))
                    tiles.append(warped)
                    final_contours.append(box) # add the coordinates of the corners of the rectangle to the list of contours
                    coords.append(box[0]) # get the coordinates of the top left corner of the rectangle
            else:
                im_draw = cv2.drawContours(im_draw, [cnt], 0, (255, 0, 0), 4) # draw contours in red color

    return final_contours, coords, im_draw, tiles

# Save the tiles from the folder "tiles":
def save_tiles(image_index, folder_image = "train2", folder ="tiles", path = "data_project"):
    # Load the image
    im = load_input_image(image_index, folder_image, path)
    # Segment the image with Canny Edge Detection
    md = segment_image_canny(im)
    # Apply Mathematical morphology
    res = apply_mat_morph(md)
    # Obtain the contours
    _, _, _, tiles = get_contours(im, res)
    num_tiles = len(tiles)
    print("Number of tiles: {}".format(num_tiles))
    # Extract the tiles as 128x128 images and save them in folder "tiles" in .png format:
    for i, tile in enumerate(tiles):
        filename = os.path.join(path, folder, "tile_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)))
        Image.fromarray(tile).save(filename)

    return tiles, num_tiles

""" FEATURES EXTRACTION """

# Get the RGB features of the tiles:
def get_hist_features(image, bins=32):
    red = np.histogram(image[:, :, 0], bins=bins)[0]
    green = np.histogram(image[:, :, 1], bins=bins)[0]
    blue = np.histogram(image[:, :, 2], bins=bins)[0]
   
    vector = np.concatenate([red, green, blue], axis=0)
    vector = vector.reshape(-1)
    return vector

# Get the gabor features of the tiles:
def compute_gabor_feats(image, kernels):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    feats = np.zeros((len(kernels), 8), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap') # mode = 'wrap' to avoid border effects
        # Mean
        feats[k, 0] = filtered.mean()
        # Variance
        feats[k, 1] = filtered.var()
        # Skewness
        feats[k, 2] = skew(filtered.flatten())
        # Kurtosis
        feats[k, 3] = kurtosis(filtered.flatten())
        # Max
        feats[k, 4] = filtered.max()
        # Min
        feats[k, 5] = filtered.min()
        # Standard deviation
        feats[k, 6] = filtered.std()
        # Power spectrum:
        spectrum = np.abs(np.fft.fft2(filtered))
        # Total power of the signal
        feats[k, 7] = np.sum(spectrum**2)
    return feats

# performs feature selection: HERE
def feature_selection(feature_map):
    """This function selects the best features from the feature map."""
    # remove features that have low variance:
    selector = VarianceThreshold()
    selector.fit(feature_map)

    # remove features that are correlated:
    #corr = np.corrcoef(feature_map, rowvar=False) # returns the correlation matrix
    #corr = np.abs(corr) # take the absolute value of the correlation matrix
    #corr = np.triu(corr, k=1)  # select only the upper triangular part of the matrix, as it is symmetric
    #corr = corr > 0.99 # returns True = 1 for the features that are highly correlated
    #corr = np.sum(corr, axis=0) == 0 # select the features that are not highly correlated
    #feature_map = feature_map[:, corr]

    return feature_map

def dimensionality_reduction(feature_maps):
    """ Reduce the dimensionality of the feature maps using PCA and choose the best # of components based on the Elbow method. """
    # flatten the feature maps:
    #feature_maps = np.array(feature_maps)
    feature_maps = feature_maps.reshape(feature_maps.shape[0], -1)
    # standardize the data:
    feature_maps = StandardScaler().fit_transform(feature_maps)
    # apply PCA and plot the Elbow method to choose the number of components:
    # pca = PCA()
    # pca.fit(feature_maps)
    #plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plot the x where the cumulative explained variance is 70%:
    #plt.axvline(x=np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.7)+1, c='r')
    #plt.xlabel('number of components')
    #plt.ylabel('cumulative explained variance')
    #plt.show()
    # print the number of components for which the cumulative explained variance is 70%:
    #print("Number of components for 70% variance: {}".format(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.7)+1))
    # apply PCA with the chosen number of components:
    # print(np.cumsum(pca.explained_variance_ratio_))
    # print(np.argmax(np.cumsum(pca.explained_variance_ratio_)>= 0.9))
    # n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_)>= 0.9)
    pca = PCA(n_components=3)
    pca.fit(feature_maps)
    feature_maps = pca.transform(feature_maps)
    # plot the results:
    # plt.figure(figsize=(5, 5))
    # plt.scatter(feature_maps[:, 0], feature_maps[:, 1], s=10)
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.show()    
    
    return feature_maps

""" CLUSTERING """

def cluster(feature_maps, n_clusters=4):
    # cluster the feature vectors using k-means and elbow method to find the optimal number of clusters:
    # normalize the feature vectors:
    #feature_maps = feature_maps - np.mean(feature_maps, axis=0)
    #feature_maps = feature_maps / np.std(feature_maps, axis=0)
    scaler = StandardScaler()
    feature_maps = scaler.fit_transform(feature_maps)
    # replace NaN values with 0:
    feature_maps = np.nan_to_num(feature_maps)
    # apply k-means clustering to the feature maps:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feature_maps)
    # get the labels of the clusters
    labels = kmeans.labels_
    # get the centroids of the clusters
    centroids = kmeans.cluster_centers_
    return labels, centroids

def plot_clusters(image_number, labels, tiles_original):
    """ This functions retrieve the tiles images according to their cluster label and plot them in a subplot"""
    # create a list of tiles for each cluster:
    clusters = []
    for i in range(len(np.unique(labels))):
        clusters.append([tiles_original[j] for j in range(len(tiles_original)) if labels[j] == i])
    # Check if there are outliers:
    flag = 0
    for i in range(len(clusters)):
        if len(clusters[i]) < 9:
            if flag == 0:
                # put the outlier as the last element of the list:
                clusters.append(clusters[i])
                clusters.pop(i)
                flag =1
                break
    # plot a subplot of the tiles for each cluster:
    for i in range(len(clusters)):
        fig, axes = plt.subplots(1, len(clusters[i]), figsize=(15, 5))
        if len(clusters[i]) < 9 and i == len(clusters)-1:
            # axes.imshow(cv2.cvtColor(clusters[i][0], cv2.COLOR_RGB2GRAY))
            # show last cluster element of the list:
            if len(clusters[i]) == 1:
                axes.imshow(clusters[i][0])
                # axes.imshow(cv2.cvtColor(clusters[i][0], cv2.COLOR_RGB2GRAY), cmap='gray')
                axes.set_title(f'Outlier of Image {image_number}')
                axes.set_axis_off()
            else: 
                for j, ax in enumerate(axes):
                    ax.imshow(clusters[i][j])
                    # ax.imshow(cv2.cvtColor(clusters[i][j], cv2.COLOR_RGB2GRAY), cmap='gray')
                    ax.set_title(f'Outlier of Image {image_number}')
                    ax.set_axis_off()
            # print("Outlier of Image {}".format(image_number))
        else :
            for j, ax in enumerate(axes.flatten()):
                ax.imshow(clusters[i][j])
                # ax.imshow(cv2.cvtColor(clusters[i][j], cv2.COLOR_RGB2GRAY), cmap='gray')
                ax.set_title(f'Cluster {i}')
                ax.set_axis_off()
            print("Image {}".format(image_number))
            plt.show()
    return clusters

def get_puzzle(img_sep):
    # put all the tiles in a single numpy array based on the size
    num_rows = len(img_sep)
    num_cols = 1
    if len(img_sep) == 9:
        num_rows = num_cols = 3
        img_sep = np.concatenate([np.concatenate(img_sep[i*num_cols:(i+1)*num_cols], axis=1) for i in range(num_rows)], axis=0)
    elif len(img_sep)==12:
        num_rows = 4
        num_cols = 3
        img_sep = np.concatenate([np.concatenate(img_sep[i*num_cols:(i+1)*num_cols], axis=1) for i in range(num_rows)], axis=0)
    elif len(img_sep)==16:
        num_rows = num_cols = 4
        # concatenate the tiles based on the number of rows and columns:
        img_sep = np.concatenate([np.concatenate(img_sep[i*num_cols:(i+1)*num_cols], axis=1) for i in range(num_rows)], axis=0)
    elif len(img_sep)>16:
        num_rows = 4
        num_cols = len(img_sep)//4
        img_sep = np.concatenate([np.concatenate(img_sep[i*num_cols:(i+1)*num_cols], axis=1) for i in range(num_rows)], axis=0)
    return img_sep