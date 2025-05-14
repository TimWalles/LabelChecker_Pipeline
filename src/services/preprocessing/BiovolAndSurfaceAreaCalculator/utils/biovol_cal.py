import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import measure
from skimage.morphology import skeletonize
from skimage.transform import rotate

def view_LC_by_ID(input_path, ID):
    """
    Get x y width height and the path to TIFF image of the image in LabelChecker*.csv file from ID.
    
    Args:
        input_path (str): Path to LabelChecker*.csv file
        ID (int): ID of the image to view
    """

    df = pd.read_csv(input_path)

    temp_df = df.loc[df['Id'] == ID, ['ImageX', 'ImageY', 'ImageW', 'ImageH', 'CollageFile']]

    if not temp_df.empty:
        x = temp_df.iloc[0]['ImageX']
        y = temp_df.iloc[0]['ImageY']
        w = temp_df.iloc[0]['ImageW']
        h = temp_df.iloc[0]['ImageH']

    source_path = os.path.dirname(input_path)
    CollageFile_path = os.path.join(source_path, temp_df.iloc[0]['CollageFile'])

    return CollageFile_path, x, y, w, h

def get_nparray_from_tiff(input_path, x, y, width, height):
    """
    Extract a region of interest from a TIFF image and return it as a numpy array.
    
    Args:
        input_path (str): Path to input TIFF image
        x (int): X coordinate of top-left corner
        y (int): Y coordinate of top-left corner
        width (int): Width of the region to extract
        height (int): Height of the region to extract
    """
    try:
        # Open the TIFF image
        with Image.open(input_path) as img:
            # Define the box coordinates (left, upper, right, lower)
            box = (x, y, x + width, y + height)
            
            # Crop the image
            img = img.convert('L')
            roi = np.array(img.crop(box))
            
            # Return nparray
            return roi
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

def get_nparray_from_png(input_path):
    """
    Extract a region of interest from a TIFF image and return it as a numpy array.
    
    Args:
        input_path (str): Path to input TIFF image
        x (int): X coordinate of top-left corner
        y (int): Y coordinate of top-left corner
        width (int): Width of the region to extract
        height (int): Height of the region to extract
    """
    
    try:
        # Open the TIFF image
        with Image.open(input_path) as img:
            
            # Crop the image
            img = img.convert('L')
            img = np.array(img)
            
            # Return nparray
            return img
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

def detect_edges(image, method='canny', low_threshold=50, high_threshold=150):
    """
    Detect edges in an RGB image using various methods.
    
    Parameters:
    image (numpy.ndarray): RGB input image
    method (str): Edge detection method ('canny', 'sobel', or 'laplacian')
    low_threshold (int): Lower threshold for Canny edge detection (only for 'canny' method)
    high_threshold (int): Higher threshold for Canny edge detection (only for 'canny' method)
    
    Returns:
    numpy.ndarray: Binary edge image (255 for edges, 0 for background)
    """
    
    if method.lower() == 'canny':
        # Canny edge detection
        edges = cv2.Canny(image, low_threshold, high_threshold)
        
    elif method.lower() == 'sobel':
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Sobel edge detection
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and convert to uint8
        edges = np.uint8(255 * magnitude / np.max(magnitude))
        
        # Apply threshold to create binary edge image
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        
    elif method.lower() == 'laplacian':
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Laplacian edge detection
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # Convert to absolute values and normalize
        edges = np.uint8(np.absolute(laplacian))
        
        # Apply threshold to create binary edge image
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        
    else:
        raise ValueError("Method must be 'canny', 'sobel', or 'laplacian'")
    
    return edges

def detect_single_neighbor_pixels(image):
    """
    Detect pixel with exactly one neighbor.
    
    Parameters:
    image (numpy.ndarray): Binary image (255 for edges, 0 for background)
    
    Returns:
    numpy.ndarray: Binary image (255 for edges, 0 for background)
    """
    # Create a kernel for 8-connectivity
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    # Apply convolution to count neighbors
    neighbor_count = ndimage.convolve(image.astype(int), kernel, mode='constant', cval=0)
    
    # Find pixels with exactly one neighbor (and that are part of the pattern)
    single_neighbor_mask = (neighbor_count == 255) & (image > 0)
    
    return single_neighbor_mask

def find_endpoints(image):
    """
    Find endpoints (pixels with exactly one neighbor) in a binary image.

    Parameters:
    image (numpy.ndarray): Binary image (255 for edges, 0 for background)

    Returns:
    list[tuple[int, int]]: List of (x, y) coordinates of endpoints
    """
    single_neighbor_pixels = detect_single_neighbor_pixels(image)
    endpoints = np.where(single_neighbor_pixels)
    return list(zip(endpoints[0], endpoints[1]))

# Function to connect endpoints with straight lines
def connect_endpoints(image, endpoints):
    """
    Connect endpoints with straight lines in a binary image.

    Parameters:
    image (numpy.ndarray): Binary image (255 for edges, 0 for background)
    endpoints (list[tuple[int, int]]): List of (x, y) coordinates of endpoints

    Returns:
    numpy.ndarray: Binary image with connected endpoints (255 for edges, 0 for background)
    """
    
    result = image.copy()
    
    if len(endpoints) < 2:
        return result
    
    # Calculate distances between all pairs of endpoints
    connected = set()  # Keep track of connected endpoints
    
    # For each endpoint, find its closest neighbor
    for i, point1 in enumerate(endpoints):
        min_dist = float('inf')
        closest_idx = -1
        
        for j, point2 in enumerate(endpoints):
            if i == j:  # Skip self
                continue
                
            # Calculate Euclidean distance
            dist = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = j
        
        # Connect this point to its closest neighbor if not already connected
        if closest_idx != -1:
            # Sort the pair to avoid duplicates
            pair = tuple(sorted([(i, closest_idx)]))
            if pair not in connected:
                start_point = (endpoints[i][1], endpoints[i][0])  # (x, y) format for cv2.line
                end_point = (endpoints[closest_idx][1], endpoints[closest_idx][0])
                cv2.line(result, start_point, end_point, 255, 1)  # White line with value 1
                connected.add(pair)
    
    return result

def reverse_floadfill(image, seed = (0,0)):

    image = cv2.copyMakeBorder(
        image,
        top=1,
        bottom=1,
        left=1,
        right=1,
        borderType=cv2.BORDER_CONSTANT,
        value=0  # Black border
    )

    ih, iw = image.shape
    mask = np.zeros((ih+2, iw+2), dtype=np.uint8)
    fill_image = image.copy()
    cv2.floodFill(fill_image, mask, seed, 255)
    fill_image = cv2.bitwise_not(fill_image)
    fill_image = image | fill_image

    h, w = fill_image.shape
    output_image = fill_image[1:h-1, 1:w-1]

    return output_image

def remove_background(image, min_size = 10):
    """
    Remove background using edge detection algorithms.
    
    Parameters:
    image (numpy.ndarray): RGB input image
    
    Returns:
    numpy.ndarray: Binary edge image
    """
    canny_edge = detect_edges(image, method='canny')

    # Skeletonize the detected edges
    labels = measure.label(canny_edge, connectivity=2)
    n_objects = np.max(labels)

    skeletionized_image = np.zeros_like(image)

    for i in range(1, n_objects + 1):
        blob_mask = np.zeros_like(image)
        blob_mask[labels == i] = 1
        skeleton = skeletonize(blob_mask)
        skeleton = skeleton.astype(np.uint8)*255
        skeletionized_image = np.maximum(skeletionized_image, skeleton)

    # Flood fill
    fill_image = reverse_floadfill(skeletionized_image)

    # Connect endpoints
    endpoints = find_endpoints(fill_image)
    connected_shape = connect_endpoints(fill_image, endpoints)

    # Flood fill again
    filled_connected_shape = reverse_floadfill(connected_shape)

    # Log the used algorithm for debug
    output = filled_connected_shape
    edge_algorithm_used = 'canny'

    # If canny didn't work, use sobel
    h, w = image.shape
    if np.count_nonzero(output) < np.ceil(0.2*h*w):
        
        edge_algorithm_used = 'sobel'
        sobel_edge = detect_edges(image, method='sobel')
        filled_sobel_edge = reverse_floadfill(sobel_edge)
        output = filled_sobel_edge
        
    return output, edge_algorithm_used

def denoise(image, min_size = 10):
    """
    Denoise a binary image by:
    1. Removing small blobs
    
    Parameters:
    image (numpy.ndarray): Binary input image (0 for black, 255 for white)
    min_size (int): Minimum number of pixels required to keep a blob
    
    Returns:
    numpy.ndarray: Denoised binary image
    """
    # Make sure the input image is binary
    if len(image.shape) > 2:
        raise ValueError("Input image must be binary (single channel)")

    image = cv2.copyMakeBorder(
        image,
        top=1,
        bottom=1,
        left=1,
        right=1,
        borderType=cv2.BORDER_CONSTANT,
        value=0  # Black border
    )

    # Step 1: Remove small blobs
    output = image.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )
    
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_size:
            output[labels == label] = 0
    
    return output

def find_perimeter(B):
    B = np.array(B).astype(np.bool_) * 1
    """find boundaries via erosion and logical and,
    using four-connectivity"""
    # return B & np.invert(binary_erosion(B,FOUR))
    S = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return ndimage.correlate(B, S, mode="constant") > 0

def check_and_fix_border_pixels(image_array):
    """Sometime the border of the blob image are all white, so we need to fix it"""
    
    # Check top and bottom rows
    top_row = image_array[0, :]
    bottom_row = image_array[-1, :]
    
    # Check left and right columns (excluding corners to avoid checking twice)
    left_col = image_array[1:-1, 0]
    right_col = image_array[1:-1, -1]
    
    # Check if all border pixels are 1
    all_borders = np.concatenate([top_row, bottom_row, left_col, right_col])

    total_elements = all_borders.size
    num_ones = np.sum(all_borders == 1)
    percentage = num_ones / total_elements

    if percentage > 0.9:
        image_array[0, :] = image_array[-1, :] = image_array[:, 0] = image_array[:, -1] = 0

    return image_array

def perimeter_image(image):
    """perimeter of blob defined via erosion and logical and"""
    img = find_perimeter(image)
    img = check_and_fix_border_pixels(img)
    return img

def perimeter_points(perimeter_image):
    """points on the perimeter of the blob"""
    return np.where(perimeter_image)

def convex_hull(perimeter_points, debug = False):
    P = np.vstack(perimeter_points).T
    hull = ConvexHull(P, qhull_options='QJ')

    if debug == True:
        points = np.vstack(perimeter_points).T
        hull = ConvexHull(points, qhull_options='QJ')

        # Plot
        plt.scatter(points[:,0], points[:,1], c='b', label='Points')

        # Plot the convex hull boundary
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'r-')

        plt.axis('equal')
        plt.legend()
        plt.show()
        
    return P[hull.vertices]

def convex_hull_properties(hull):
    """compute perimeter and area of convex hull"""
    ab = hull - np.roll(hull, 1, axis=0)
    # first compute length of each edge
    C = np.sqrt(np.sum(ab**2, axis=1))
    # perimeter is sum of these
    perimeter = np.sum(C)
    # compute distance from center to each vertex
    center = np.mean(hull, axis=0)
    A = np.sqrt(np.sum((hull - center) ** 2, axis=1))
    # A, B, C will now be all triangle edges
    B = np.roll(A, 1, axis=0)
    # heron's forumla requires s = half of each triangle's perimeter
    S = np.sum(np.vstack((A, B, C)), axis=0) / 2
    # compute area of each triangle
    areas = np.sqrt(S * (S - A) * (S - B) * (S - C))
    # area of convex hull is sum of these
    area = np.sum(areas)
    # add half-pixel adjustment for each unit distance along
    # perimeter to adjust for rasterization
    area += perimeter / 2
    return perimeter, area

def area_cal(image):
    return measure.regionprops(image.astype(np.uint8))[0].area

def calculate_equivalent_diameter(area):
    """
    Calculate equivalent diameter from an area value
    Formula: diameter = sqrt(4 * area / π)
    """
    diameter = np.sqrt(4 * area / np.pi)
    return diameter

def ellipse_properties(B):
    """returns major axis length, minor axis length, eccentricity,
    and orientation"""
    """note that these values are all computable using
    skimnage.measure.regionprops, which differs only in that
    it returns the orientation in radians"""
    P = np.vstack(np.where(B))  # coords of all points
    # magnitudes and orthonormal basis vectors
    # are computed via the eigendecomposition of
    # the covariance matrix of the coordinates
    (
        eVal,
        eVec,
    ) = eig(np.cov(P))
    # axes lengths are 4x the sqrt of the eigenvalues,
    # major and minor lenghts are max, min of them
    L = 4 * np.sqrt(eVal)
    maj_axis, min_axis = np.max(L), np.min(L)

    # orientation is derived from the major axis's
    # eigenvector

    x, y = eVec[:, np.argmax(L)]
    orientation = (180 / np.pi) * np.arctan(y / x) - 90

    # eccentricity = 1st eccentricity
    ecc = np.sqrt(1 - (min_axis / maj_axis) ** 2)

    return maj_axis, min_axis, ecc, orientation

def center_blob(B):
    """returns a new image centered on the blob's
    centroid"""
    # compute centroid
    yc, xc = np.mean(np.vstack(np.where(B)), axis=1)
    # center
    h, w = B.shape
    s = max(yc, h - yc, xc, w - xc)
    m = int(np.ceil(s * 2))
    C = np.zeros((m, m), dtype=np.bool_)
    y0, x0 = int(np.floor(s - yc)), int(np.floor(s - xc))
    C[y0 : y0 + h, x0 : x0 + w] = B
    return C

def rotate_blob(blob, theta):
    """rotate a blob counterclockwise"""
    blob = center_blob(blob)
    # note that v2 uses bilinear interpolation in MATLAB
    # and that is not available in skimage rotate
    # so v3 uses nearest-neighbor
    blob = rotate(blob, -1 * theta, order=0).astype(np.bool_)
    # note that v2 does morphological post-processing and v3 does not
    return blob

def bottom_top_area(X, Y, Z, ignore_ground=False):
    """computes top quad and bottom quad areas for distmap
    and SOR algorithms"""
    """ignore_ground is an adjustment used in distmap
    but not in SOR"""
    h, w = Z.shape

    i2 = slice(0, h - 1)
    i1 = slice(1, h)
    ia2 = slice(0, w - 1)
    ia1 = slice(1, w)

    # create linesegs AB for all quadrilaterals
    AB1, AB2, AB3 = [xyz[i2, ia2] - xyz[i1, ia2] for xyz in [X, Y, Z]]
    # create linesegs AD for all quadrilaterals
    AD1, AD2, AD3 = [xyz[i2, ia2] - xyz[i1, ia1] for xyz in [X, Y, Z]]
    # create linesegs AD for all quadrilaterals
    CD1, CD2, CD3 = [xyz[i2, ia1] - xyz[i1, ia1] for xyz in [X, Y, Z]]

    # triangle formed by AB and AD for all quadrilaterals
    leg1 = ((AB2 * AD3) - (AB3 * AD2)) ** 2
    leg2 = ((AB3 * AD1) - (AB1 * AD3)) ** 2
    leg3 = ((AB1 * AD2) - (AB2 * AD1)) ** 2
    # bottom area
    area_bot = 0.5 * np.sqrt(leg1 + leg2 + leg3)

    # triangle formed by CD and AD for all quadrilaterals
    leg1 = ((CD2 * AD3) - (CD3 * AD2)) ** 2
    leg2 = ((CD3 * AD1) - (CD1 * AD3)) ** 2
    leg3 = ((CD1 * AD2) - (CD2 * AD1)) ** 2
    # top area
    area_top = 0.5 * np.sqrt(leg1 + leg2 + leg3)

    if ignore_ground:
        ind = np.abs(AB3) + np.abs(AD3) + np.abs(CD3) + Z[i2, ia2]
        area_bot[ind == 0] = 0
        area_top[ind == 0] = 0

    return area_bot, area_top

def sor_volume_surface_area(B):
    """pass in rotated blob"""
    """Sosik and Kilfoyle surface area / volume algorithm"""
    # find the bottom point of the circle for each column
    m = np.argmax(B, axis=0)
    # compute the radius of each slice
    d = np.sum(B, axis=0)
    # exclude 0s
    r = (d / 2.0)[d > 0]
    m = m[d > 0]
    n_slices = r.size
    # compute 721 angles between 0 and 180 degrees inclusive, in radians
    n_angles = 721
    angR = np.linspace(0, 180, n_angles) * (np.pi / 180)

    # make everything the same shape: (nslices, nangles)
    angR = np.vstack([angR] * m.size)
    m = np.vstack([m] * n_angles).T
    r = np.vstack([r] * n_angles).T

    # compute the center of each slice
    center = m + r
    # correct for edge effects
    center[0, :] = center[1, :]
    center[-1, :] = center[-2, :]

    # y coordinates of all angles on all slices
    Y = center + np.cos(angR) * r
    # z coordinates of all angles on all slices
    Z = np.sin(angR) * r

    # compute index of slice in y matrix
    x = np.array(range(r.shape[0])) + 1.0
    # half-pixel adjustment of edges
    x[0] -= 0.5
    x[-1] += 0.5
    X = np.vstack([x] * n_angles).T

    # compute bottom and top area
    area_bot, area_top = bottom_top_area(X, Y, Z)

    # surface area
    # multiply sum of areas of quadrilaterals by 2 to account for angles 180-360
    sa = 2 * (np.sum(area_bot) + np.sum(area_top))
    # add flat end caps
    sa += np.sum(np.pi * r[[0, -1], 0] ** 2)

    # compute height of cone slices
    b1 = np.pi * r[1:n_slices, 0] ** 2
    b2 = np.pi * r[0 : n_slices - 1, 0] ** 2
    h = np.diff(x)
    # volume
    v = np.sum((h / 3) * (b1 + b2 + np.sqrt(b1 * b2)))

    # representative width
    xr = np.mean(r[:, 0] * 2)

    # return volume, representative width, and surface area
    return v, xr, sa

def distmap_volume_surface_area(B, perimeter_image=None):
    """Moberg & Sosik biovolume algorithm
    returns volume and representative transect"""
    if perimeter_image is None:
        perimeter_image = find_perimeter(B)
    # elementwise distance to perimeter + 1
    D = ndimage.distance_transform_edt(1 - perimeter_image) + 1
    # mask distances outside blob
    D = D * (B > 0)
    Dm = np.ma.array(D, mask=1 - B)
    # representative transect
    x = 4 * np.ma.mean(Dm) - 2
    # diamond correction
    c1 = x**2 / (x**2 + 2 * x + 0.5)
    # circle correction
    # c2 = np.pi / 2
    # volume = c1 * c2 * 2 * np.sum(D)
    volume = c1 * np.pi * np.sum(D)

    distmap_img = Dm.copy()
    distmap_img = c1*np.pi*distmap_img
    distmap_img = (distmap_img - np.min(distmap_img)) / (np.max(distmap_img) - np.min(distmap_img))
    distmap_img = (distmap_img * 255).astype(np.uint8)
    distmap_img = distmap_img.filled(0)

    # surface area
    h, w = D.shape
    Y, X = np.mgrid[0:h, 0:w]
    area_bot, area_top = bottom_top_area(X, Y, D, ignore_ground=True)
    # final correction of the diamond cross-section
    # inherent in the distance map to be circular instead
    c = (np.pi * x / 2) / (2 * np.sqrt(2) * x / 2 + (1 + np.sqrt(2)) / 2)
    sa = 2 * c * (np.sum(area_bot) + np.sum(area_top))
    # return volume, representative transect, and surface area
    return volume, x, sa, distmap_img

def pixel_to_micrometer_volumn(pixel_volumn, calibration_const=2.74): # Parameters calculated separately
    return pixel_volumn*calibration_const

def preprocessing(image):
    removeBG_image, edge_algorithm_used = remove_background(image)
    clean_binary_image = denoise(removeBG_image)

    return clean_binary_image, edge_algorithm_used

def biovolume(image, calibration_const=2.74, debug = False): # Assumed grayscale image
    # Remove background
    
    clean_binary_image, edge_algorithm_used = preprocessing(image)
    # All these is to check if the blob look likes a circle
    labels = measure.label(clean_binary_image, connectivity=2)
    n_objects = np.max(labels)

    biovol_result = 0
    surface_area_result = 0
    result_mask = np.zeros_like(clean_binary_image)
    formular_used = []
    biovol_each_blob = []
    for i in range(1, n_objects + 1):
        try:
            blob_mask = np.zeros_like(clean_binary_image)
            blob_mask[labels == i] = 1

            # Criteria 1
            perimeter_image_var = perimeter_image(blob_mask)
            perimeter_points_var = perimeter_points(perimeter_image_var)
            convex_hull_var = convex_hull(perimeter_points_var)
            _, convex_area = convex_hull_properties(convex_hull_var)
            area = area_cal(blob_mask)
            area_ratio = float(convex_area) / area

            # Criteria 2
            eccentricity = ellipse_properties(blob_mask)[2]

            # Criteria 3
            equiv_diameter = calculate_equivalent_diameter(area)
            major_axis_length = ellipse_properties(blob_mask)[0]
            p = equiv_diameter / major_axis_length

            orientation = (180 / np.pi) * measure.regionprops(blob_mask.astype(np.uint8))[0].orientation
            rotated_image =  rotate_blob(blob_mask, orientation)

            if area_ratio < 1.2 or (eccentricity < 0.8 and p > 0.8):

                sor = sor_volume_surface_area(rotated_image)
                biovol_result = biovol_result + pixel_to_micrometer_volumn(sor[0], calibration_const)
                surface_area_result = surface_area_result + pixel_to_micrometer_volumn(sor[2], calibration_const)

                vol_to_2d_area_ratio = int(sor[0]/area)
                result_mask = np.maximum(result_mask, blob_mask*vol_to_2d_area_ratio)

                # log for debug
                formular_used.append('Sor')
                biovol_each_blob.append(pixel_to_micrometer_volumn(sor[0], calibration_const))

            else:
                perimeter_image_var = perimeter_image(blob_mask)
                distmap_result = distmap_volume_surface_area(blob_mask, perimeter_image_var)
                biovol_result = biovol_result + pixel_to_micrometer_volumn(distmap_result[0], calibration_const)
                surface_area_result = surface_area_result + pixel_to_micrometer_volumn(distmap_result[2], calibration_const)

                result_mask = np.maximum(result_mask, distmap_result[3])

                # log for debug
                formular_used.append('Distmap')
                biovol_each_blob.append(pixel_to_micrometer_volumn(distmap_result[0], calibration_const))

        except Exception as e:
            #print(f"Error processing some blob: {str(e)}")
            biovol_each_blob.append(0)
            continue

    if debug == True:
        print(f"Edge detection algorithm used : {edge_algorithm_used}")
        print(f"Formular used : {formular_used}")
        print(f"Biovol for each blob : {biovol_each_blob}")
        print(f"Total biovol : {biovol_result}")
        print(f"Total surface area : {surface_area_result}")
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(4, 1, figsize=(20, 24), dpi=150)  # 1 row, 3 columns
        # Display images in each subplot
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Input image')
        axes[0].axis('off')  # Turn off axis numbers

        axes[1].imshow(clean_binary_image, cmap='gray')
        axes[1].set_title('Preprocessed')
        axes[1].axis('off')

        axes[2].imshow(labels, cmap='gray')
        axes[2].set_title('Blobs')
        axes[2].axis('off')

        axes[3].imshow(result_mask)
        axes[3].set_title('Final result')
        axes[3].axis('off')

        # Add space between subplots
        plt.tight_layout()

        # Display the figure
        plt.show()

    return result_mask, round(biovol_result,2), round(surface_area_result,2)