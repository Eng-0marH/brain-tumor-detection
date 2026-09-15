import cv2

from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE


def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def pad_and_clip(x1, y1, x2, y2, width, height, padding_ratio):
    """Grow a box by padding_ratio on each side and keep it inside the image.

    Used for both the ground-truth crops in classifier_prep and the predicted
    crops at inference time, so the classifier always sees the same framing.
    """
    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)

    x1 = max(0, min(x1 - pad_x, width))
    y1 = max(0, min(y1 - pad_y, height))
    x2 = max(0, min(x2 + pad_x, width))
    y2 = max(0, min(y2 + pad_y, height))

    return x1, y1, x2, y2
