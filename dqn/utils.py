import numpy as np
from PIL import Image

from config import Config

def preprocess_frame(frame: np.ndarray) -> np.ndarray:

    img = Image.fromarray(frame).convert("L").resize(
        (Config.FRAME_SIZE, Config.FRAME_SIZE), Image.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32)
    arr[arr > 1.0] = 255.0
    arr[arr <= 1.0] = 0.0
    return arr

def init_state(frame: np.ndarray) -> np.ndarray:
    f = preprocess_frame(frame)
    return np.stack([f] * Config.STACK_LEN, axis=0)

def next_state(state: np.ndarray, frame: np.ndarray) -> np.ndarray:
    f = preprocess_frame(frame)
    return np.concatenate([f[None, :, :], state[:Config.STACK_LEN - 1]], axis=0)
