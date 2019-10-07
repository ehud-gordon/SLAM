import os
import cv2
import numpy as np


# ============================================
# Camera calibration
# ============================================
def chess_calibrate(cal_images_dirpath, res):
    # Define size of chessboard target.
    width, height = 9, 6
    pattern_size = (width, height)
    # Define arrays to save detected points
    obj_points = []  # 3D points in real world space
    img_points = []  # 3D points in image plane

    # Prepare grid and points to display
    pattern_points = np.zeros((width * height, 3), dtype=np.float32)
    pattern_points[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    # read images
    num_of_detections, num_of_images = 0, 0
    # get height and width of images. All images should be of same width and height
    im_name = os.listdir(cal_images_dirpath)[0]
    im = cv2.imread(os.path.join(cal_images_dirpath, im_name), cv2.IMREAD_GRAYSCALE)
    h,w = im.shape[:2]

    # Iterate over images to find intrinsic matrix
    for image_name in os.listdir(cal_images_dirpath):

        # Load image
        img = cv2.imread(os.path.join(cal_images_dirpath, image_name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Make sure images are horizontal, not vertical
        if img.shape[0] != h or img.shape[1] != w:
            print("{} of wrong dimensions".format(image_name))
            continue

        # find chessboard corners
        num_of_images += 1
        found, corners = cv2.findChessboardCorners(img, pattern_size)

        if found:
            print("Chessboard detected! " + image_name)
            num_of_detections+=1
            # define criteria for subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            # refine corner location (to subpixel accuracy) based on criteria.
            cv2.cornerSubPix(image=img, corners=corners, winSize=(5, 5), zeroZone=(-1, -1),
                             criteria=criteria)
            obj_points.append(pattern_points)
            img_points.append(corners)
            # Uncomment this in order to save images of found corners
            # image_of_drawn_corners = img.copy()
            # cv2.drawChessboardCorners(image_of_drawn_corners, (9,6), corners, ret)
            # basename = str(os.path.basename(image_name).split(".")[0])
            # cv2.imwrite("found_{}_{}.png".format(basename, res), image_of_drawn_corners)

        else:
            print("Chessboard not detected: " + image_name)

    print("{}\{} images detected".format(num_of_detections, num_of_images))

    if len(img_points) == 0:
        print("couldn't find any chessboards")
        return

    save_camera_params(obj_points, img_points, (w,h), str(res))



def save_camera_params(obj_points, img_points, size, res):
    # Calibrate camera
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points,img_points, imageSize=size,
                                                     cameraMatrix=None, distCoeffs=None)
    
    # Save parameters into numpy files
    param_dir = "camera_params_" + res
    os.makedirs(param_dir, exist_ok=True)
    np.save(file=os.path.join(param_dir, 'rms'), arr=rms)
    np.save(file=os.path.join(param_dir, 'K'), arr=K)
    np.save(file=os.path.join(param_dir, 'dist'), arr=dist)
    np.save(file=os.path.join(param_dir, 'rvecs'), arr=rvecs)
    np.save(file=os.path.join(param_dir, 'tvecs'), arr=tvecs)
    
    # print overall RMS reprojection error
    print("Re-projection error reported by calibrateCamera: {}".format(rms))
    

if __name__ == '__main__':
    res = '480'
    cal_images_dirpath = "cal_images_" + res
    chess_calibrate(cal_images_dirpath=cal_images_dirpath, res=res)
