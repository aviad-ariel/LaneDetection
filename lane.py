import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
from importlib import reload
import utils; reload(utils)
from utils import *
from classes import *
from moviepy.editor import VideoFileClip


def findChessboardCorners(img, nx, ny):
    """
    Finds the chessboard corners of the supplied image (must be grayscale)
    nx and ny parameters respectively indicate the number of inner corners in the x and y directions
    """
    return cv2.findChessboardCorners(img, (nx, ny), None)


def findImgObjPoints(imgs_paths, nx, ny):
    """
    Returns the objects and image points computed for a set of chessboard pictures taken from the same camera
    nx and ny parameters respectively indicate the number of inner corners in the x and y directions
    """
    objpts = []
    imgpts = []
    
    # Pre-compute what our object points in the real world should be (the z dimension is 0 as we assume a flat surface)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    
    for img_path in imgs_paths:
        img = load_image(img_path)
        gray = to_grayscale(img)
        ret, corners = findChessboardCorners(gray, nx, ny)
        
        if ret:
            # Found the corners of an image
            imgpts.append(corners)
            # Add the same object point since they don't change in the real world
            objpts.append(objp)
    
    return objpts, imgpts


def undistort_image(img, objpts, imgpts):
    """
    Returns an undistorted image
    The desired object and image points must also be supplied to this function
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, to_grayscale(img).shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def threshold_img(img, channel, thres=(0, 255)):
    """
    Applies a threshold mask to the input image
    """
    img_ch = img[:,:,channel]
    if thres is None:  
        return img_ch
    
    mask_ch = np.zeros_like(img_ch)
    mask_ch[ (thres[0] <= img_ch) & (thres[1] >= img_ch) ] = 1
    return mask_ch


def compute_perspective_transform_matrices(src, dst):
    """
    Returns the tuple (M, M_inv) where M represents the matrix to use for perspective transform
    and M_inv is the matrix used to revert the transformed image back to the original one
    """
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    return (M, M_inv)


def perspective_transform(img, src, dst):   
    """
    Applies a perspective 
    """
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped


if __name__ == "__main__":
    calibration_dir = "camera_cal"
    test_imgs_dir = "test_images"
    output_imgs_dir = "output_images"
    output_videos_dir = "output_videos"
    cx = 9
    cy = 6
    cal_imgs_paths = glob.glob(calibration_dir + "/*.jpg")
    opts, ipts = findImgObjPoints(cal_imgs_paths, cx, cy)
    test_imgs_paths = glob.glob(test_imgs_dir + "/*.jpg")
    test_imgs = np.asarray(list(map(lambda img_path: load_image(img_path), test_imgs_paths)))
    undist_test_imgs = np.asarray(list(map(lambda img: undistort_image(img, opts, ipts), test_imgs)))
    copy_combined = np.copy(undist_test_imgs[1])
    (bottom_px, right_px) = (copy_combined.shape[0] - 1, copy_combined.shape[1] - 1) 
    pts = np.array([[400,bottom_px],[1000,600],[1150,600], [1810, bottom_px]], np.int32)
    src_pts = pts.astype(np.float32)
    dst_pts = np.array([[200, bottom_px], [200, 0], [1000, 0], [1000, bottom_px]], np.float32)

    challenge_video_sample_path = 'lane_Trim.mp4'

    challenge_video_sample_output_path = 'output_videos/lane_Trim_new_ROI.mp4'
  
    detector = AdvancedLaneDetectorWithMemory(opts, ipts, src_pts, dst_pts, 20, 100, 10)

    clip1 = VideoFileClip(challenge_video_sample_path)
    challenge_video_clip = clip1.fl_image(detector.process_image) #NOTE: this function expects color images!!
    challenge_video_clip.write_videofile(challenge_video_sample_output_path, audio=False)
