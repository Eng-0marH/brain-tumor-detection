import cv2

from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE


def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
