# This is for post-hoc white balancing.


import cv2
import numpy as np


def apply_manual_white_balance(image: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    """
    Applies per-channel white balance to an RGB image.
    `gains` is a tuple (R_gain, G_gain, B_gain).
    """
    r_gain, g_gain, b_gain = gains
    balanced = image.astype(np.float32)
    balanced[..., 0] *= r_gain  # R channel
    balanced[..., 1] *= g_gain  # G channel
    balanced[..., 2] *= b_gain  # B channel

    # Clip to [0, 255] and convert back to uint8
    return np.clip(balanced, 0, 255).astype(np.uint8)


# Alternative white balancing methods using OpenCV


def get_gray_world_closure(initial_image):
    """
    Given an image, computes the gray world coefficients.
    Instead of applying them and returning an image, it instead
    saves them in a closure. This closure is a function : Image -> Image that
    applies the coefficients learned from the initial image.
    Note that the variable names suggest BGR images, but in reality this
    works for any ordering of the channels.
    """
    initial_image = initial_image.copy().astype(np.float32)
    avg_b = np.mean(initial_image[..., 0])
    avg_g = np.mean(initial_image[..., 1])
    avg_r = np.mean(initial_image[..., 2])

    # Calculate the average gray value
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # Calculate scaling factors
    b_scale = avg_gray / avg_b
    g_scale = avg_gray / avg_g
    r_scale = avg_gray / avg_r

    # Now generate the closure:

    def apply_fixed_gray_world(image, my_b_scale=b_scale, my_g_scale=g_scale, my_r_scale=r_scale):
        result = image.copy().astype(np.float32)
        result[..., 0] *= my_b_scale
        result[..., 1] *= my_g_scale
        result[..., 2] *= my_r_scale
        return np.clip(result, 0, 255).astype(np.uint8)

    return apply_fixed_gray_world


def apply_gray_world(image: np.ndarray) -> np.ndarray:
    """
    Applies Gray World assumption for white balancing.
    Assumes the average color in a scene should be gray.
    """
    result = image.copy().astype(np.float32)
    avg_b = np.mean(result[..., 0])
    avg_g = np.mean(result[..., 1])
    avg_r = np.mean(result[..., 2])

    # Calculate the average gray value
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # Calculate scaling factors
    b_scale = avg_gray / avg_b
    g_scale = avg_gray / avg_g
    r_scale = avg_gray / avg_r

    # Apply scaling
    result[..., 0] *= b_scale
    result[..., 1] *= g_scale
    result[..., 2] *= r_scale

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_opencv_white_balance(image: np.ndarray) -> np.ndarray:
    """
    Uses OpenCV's built-in white balance algorithm (CLAHE).
    Applies contrast limited adaptive histogram equalization to each channel.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to L channel
    lab[..., 0] = clahe.apply(lab[..., 0])

    # Convert back to RGB
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
