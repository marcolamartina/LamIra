import freenect
import cv2
import os
import numpy as np

cv2.namedWindow('Depth')
cv2.namedWindow('Video')
print('Press ESC in window to stop')


def pretty_depth(depth):
    """Converts depth into a 'nicer' format for display

    This is abstracted to allow for experimentation with normalization

    Args:
        depth: A numpy array with 2 bytes per pixel

    Returns:
        A numpy array that has been processed with unspecified datatype
    """
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth


def pretty_depth_cv(depth):
    """Converts depth into a 'nicer' format for display

    This is abstracted to allow for experimentation with normalization

    Args:
        depth: A numpy array with 2 bytes per pixel

    Returns:
        A numpy array with unspecified datatype
    """
    return pretty_depth(depth)


def video_cv(video):
    """Converts video into a BGR format for display

    This is abstracted out to allow for experimentation

    Args:
        video: A numpy array with 1 byte per pixel, 3 channels RGB

    Returns:
        A numpy array with with 1 byte per pixel, 3 channels BGR
    """
    return video[:, :, ::-1]  # RGB -> BGR

def get_depth():
    return pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return video_cv(freenect.sync_get_video()[0])

d=get_depth()
v=get_video()

path=os.path.dirname(__file__)
depth_file=path+"/depth.npy"
image_file=path+"/image.npy"

np.save(depth_file, d) # save
np.save(image_file, v) # save

depth = np.load(depth_file) # load
image = np.load(image_file) # load

cv2.imshow('Depth', depth)
cv2.imshow('Video', image)
cv2.waitKey(0)