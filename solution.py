'''
Submission for OCBC AI Lab Tech Interview
141019
Luar Shui Song


Versions used:
OpenCV 4.1.1
Numpy 1.16.2
Shapely 1.6.4
Matplotlib 3.1.1
Python 3.7
'''

# USER-INSTALLED LIBRARIES
import cv2
import numpy as np
from shapely.geometry import LineString, mapping
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# PYTHON NATIVE
from itertools import groupby, combinations
from operator import itemgetter
import ast
import math
import pprint 

def pre_processing(filename):
    # Credit: https://medium.com/coinmonks/a-box-detection-algorithm-for-any-image-containing-boxes-756c15d7ed26
    # Used to segment the vertical and horizontal lines from image (better than Hough Transform/Canny)

    img = cv2.imread(filename, 0)
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image

    cv2.imwrite("Image_bin.jpg",img_bin)

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40
     
    # A vertical kernel of (1 X kernel_length), which will detect all the vertical lines from the image.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=3)
    cv2.imwrite("vertical_lines.jpg",vertical_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)
    
    return horizontal_lines_img, vertical_lines_img


def counter(lines_img):
# Code to find all subsequences of '255's and put them in 3 dictionaries
# E.g. pixel detected sequence can be like this [1,2,3,255,256,257,258,259]
# This denotes that pixels 1,2,3 and 255,256,257,258,259 at that iteration (height = idx) are == 255
# Desired output: [1,2,3], [255,255,256,257,258,259]
# We further classify them into three different dictionaries:
# first_observed: first seen; dup_first_observed: to store duplicates; last_observed: last seen 
    seq_list = []
    first_observed = {}
    last_observed = {}
    dup_first_observed = {}
    # We are processing the image row by row to detect the lines
    for idx, array in enumerate(lines_img):
        summer = sum(array)
        
        # if the row is all 0's, skip
        if summer == 0: 
            continue
        else:
            subseq = np.where(array == 255)[0].tolist()
            
            for k, g in groupby(enumerate(subseq), lambda ix : ix[0] - ix[1]):
                seq_list += [list(map(itemgetter(1), g))]
                        
            for seq in seq_list:
                seq = str(seq)
                if seq not in first_observed:
                    first_observed[seq] = idx
                # Records the cases where the sequences are repeated and non-unique
                if seq not in dup_first_observed:
                    dup_first_observed[seq] = [idx]
                elif seq in dup_first_observed:
                    dup_first_observed[seq].append(idx)
                
                # always record last_observed
                last_observed[seq] = idx
  
            seq_list = []
    return first_observed, last_observed, dup_first_observed
    
# hori and vert line segments
def find_segments(combined):
    vert_seg = []
    hori_seg = []

    x_elems = [x[0] for x in combined]
    y_elems = [x[1] for x in combined]
    
    # to remove duplicates
    x_elems = list(dict.fromkeys(x_elems))
    y_elems = list(dict.fromkeys(y_elems))
    
    # find all same x coordinates ==> vertical segments
    for x in x_elems:
        temp_list = []
        for coord in combined:
            if x not in coord:
                continue
            temp_list += [coord]
            
        # order the temp_list and pick the ends appropriately
        temp_elems = [elems[1] for elems in temp_list]
        temp_elems = sorted(temp_elems)
        vert_seg += [[[x, temp_elems[0]],[x, temp_elems[-1]]]]
    
    # find all same y coordinates ==> horizontal segments
    for y in y_elems:
        temp_list = []
        for coord in combined:
            if y not in coord:
                continue
            temp_list += [coord]
            
        # order the temp_list and pick the ends appropriately
        temp_elems = [elems[0] for elems in temp_list]
        temp_elems = sorted(temp_elems)
        
        hori_seg += [[[temp_elems[0],y],[temp_elems[-1],y]]]
    
    return vert_seg, hori_seg

def angle(vector1, vector2):
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    # avoid division by zero errors
    if len1 == 0 or len2 == 0:
        return 0
    try:
        result = math.acos(inner_product/(len1*len2))
        return result
    except: 
        return 0

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

# Now time to find boxes
def box_detector(combined):
    # find any 4-tuple of coordinates within the combined data
    
    # requirement: 4 angles made by this 4-tuple has to make 90 degrees
    box_list = []
    iter = 0
    total_iters = nCr(len(combined), 4)
    
    
    for a,b,c,d in combinations(combined, 4):
        iter += 1
        print('iteration %d out of %d' %(iter, total_iters))
        line_combinations = [[a,b],[a,c],[a,d],[b,c],[b,d],[c,d]]
        count = 0
        for line1, line2 in combinations(line_combinations,2):
            #unpack values
            v1_x = line1[0][0] -line1[1][0]
            v1_y = line1[0][1] -line1[1][1]
            v2_x = line2[0][0] -line2[1][0]
            v2_y = line2[0][1] -line2[1][1] 
            if np.isclose(angle((v1_x,v1_y),(v2_x, v2_y)), np.pi/2, rtol=1e-05, atol=1e-08, equal_nan=False):
                count += 1
            if count == 4:
                box_list.append([a,b,c,d])
                continue
    

    return box_list

def basic_detection(filename):
    # Detecting Horizontal coordinates
    horizontal_lines_img, vertical_lines_img = pre_processing(filename)    
    first_observed, last_observed, dup_first_observed = counter(horizontal_lines_img)
    hori_lines = []
    hori_ycoords = []
    hori_coords = []

    # to add into our lists horizontal y coordinates and horizontal line candidates
    for items in dup_first_observed:
        hori_ycoords.append(dup_first_observed[items])
        items = ast.literal_eval(items)
        hori_lines.append(items)

    # detect consecutive numbers ==> these form a contiguous stretch
    # e.g. a detected horizontal contiguous array could look like this 
    # [30,31,32,33,......, 855,856,857] 
    # This implies that pixels 30 to 857 form a contiguous stretch of 'line',
    # therefore we want to extract the first and last point. 
    # The y-coordinate is just the same throughout.

    seq_list = []
    for idx, coords in enumerate(hori_ycoords):
        for k, g in groupby(enumerate(coords), lambda ix : ix[0] - ix[1]):
            seq_list += [list(map(itemgetter(1), g))]
        # take the first coordinate as the horizontal point
        for seq in seq_list:
            y1 = seq[0] 
            x0 = hori_lines[idx][0]
            x1 = hori_lines[idx][-1]
            
            hori_coords.append([x0,y1])
            hori_coords.append([x1,y1])
        
        seq_list = []

    # Detecting vertical coordinates
    # E.g. array could be like [30, 31, 32, 33, 34, 35]
    # Picture this as the width of the line - and therefore we can see easily that 
    # this same array would be repeated for however many rows in the image.
    # hence our x-coordinates will be the same, but y-coordinates are taken from the
    # first_observed and last_observed dictionaries respectively.
    first_observed, last_observed, _ = counter(vertical_lines_img)
    vert_lines = []
    for items in first_observed:
        items = ast.literal_eval(items)
        vert_lines.append(items)
    vert_coords = []
    for line in vert_lines:
        x0 = line[0]
        line = str(line)
        y0 = first_observed[line]
        y1 = last_observed[line]
        vert_coords.append([x0,y0])
        vert_coords.append([x0,y1])

    combined =  vert_coords+ hori_coords

    # To remove the duplicates that are close to each other. 
    coords_to_remove = []
    coords_to_add = []
    threshold = 8
    for a, b in combinations(combined, 2):
        if a == b:
            coords_to_add.append(a)
        # L2 norm
        if ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5 < threshold:
            # Keep the biggest one
            if a[0] + a[1] > b[0] + b[1]:
                coords_to_remove.append(b)
            else:
                coords_to_remove.append(a)
                
    combined = [desired for desired in combined if desired not in coords_to_remove]
    combined += coords_to_add

            
    # To fine-tune the coordinates to be on the same gridlines
    for a, b in combinations(combined, 2):
        if abs(a[0]-b[0])< threshold:
            # always keep the biggest one
            if a[0] > b[0]:
                idx = combined.index(b)
                combined[idx][0] = a[0]

            else:
                idx = combined.index(a)
                combined[idx][0] = b[0]

        if abs(a[1]-b[1]) < threshold:
            if a[1] > b[1]:
                idx = combined.index(b)
                combined[idx][1] = a[1]

            else:
                idx = combined.index(a)
                combined[idx][1] = b[1]
    return combined
        

def intersection_finder(combined):
    # to find intersection between horizontal and vertical lines
    intersection_list = []

    vert_seg, hori_seg = find_segments(combined)

    for a in vert_seg:
        for b in hori_seg:
            line1 = LineString([tuple(a[0]), tuple(a[1])])
            line2 = LineString([tuple(b[0]), tuple(b[1])])
            result = line1.intersection(line2)
            
            if not result: 
                continue

                
            result = np.array(result)
            print(result)
            intersection_list.append(result.tolist())
    print('Final combined', intersection_list, len(intersection_list))

    # To combine with the original points
    intersection_list = [[int(elem[0]),int(elem[1])] for elem in intersection_list]
    combined = intersection_list + combined
    combined = [str(temp) for temp in combined]
    combined = list(dict.fromkeys(combined))
    combined = [ast.literal_eval(temp) for temp in combined]
    return combined


def main(filename):
	'''
	# For debugging, will show the positive result
	for pts in combined:
		plt.imshow(img_bin)
		plt.scatter(pts[0], pts[1])
		
	plt.show()
	'''
	combined = basic_detection(filename)
	combined = intersection_finder(combined)
	box_list = box_detector(combined)

	'''
	# insert printing function for boxes here.
	for boxes in box_list:
		for box in boxes:
			plt.plot(box[0],box[1], 'ro')
	plt.show()
	'''

	# final step - identify duplicates
	box_list = [str(box) for box in box_list]
	box_list = list(dict.fromkeys(box_list))
	box_list = [ast.literal_eval(temp) for temp in box_list]
	pp = pprint.PrettyPrinter(indent = 4)
	pp.pprint(box_list)
	
	print('Number of Boxes', len(box_list))

if __name__ == '__main__':
	main('boxes2.jpg')