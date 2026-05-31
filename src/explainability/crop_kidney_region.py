import numpy as np


def crop_center(image, crop_ratio=0.7):
    """
    Crop the central region of an image.

    Parameters:
    - image (numpy.ndarray): Grayscale CT image
    - crop_ratio (float): Fraction of image to retain (0.6–0.75 recommended)

    Returns:
    - Cropped image
    """
    h, w = image.shape

    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)

    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    return image[start_h:start_h + crop_h,
                 start_w:start_w + crop_w]


def orientation_aware_crop(image, crop_ratio=0.7):
    """
    Apply cropping ONLY to axial CT images.
    Skip cropping for coronal/sagittal CT images.

    Axial CTs are approximately square.
    Coronal CTs are tall or rectangular.

    Parameters:
    - image (numpy.ndarray): Grayscale CT image
    - crop_ratio (float): Fraction of image to retain

    Returns:
    - Cropped image (axial) OR original image (coronal)
    """
    h, w = image.shape
    aspect_ratio = h / w

    # Axial CT images are close to square
    if 0.9 <= aspect_ratio <= 1.1:
        return crop_center(image, crop_ratio)
    else:
        return image
