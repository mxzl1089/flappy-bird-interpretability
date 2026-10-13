import os
import torch

class Config:

    NUM_ACTIONS = 2
    FRAME_SIZE = 80
    STACK_LEN = 4

    GAMMA = 0.99
    LR = 1e-4

    BATCH_SIZE = 32
    REPLAY_SIZE = 50000
    OBSERVE = 100000

    EXPLORE = 1000000
    INIT_EPSILON = 0.1
    FINAL_EPSILON = 0.01
    TARGET_SYNC = 10000
    USE_TARGET_NET = True

    MAX_STEPS = 2000000
    SAVE_EVERY = 10000
    LOG_EVERY = 1000
    USE_TB = True

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CKPT_DIR = os.path.join(ROOT, "runs")
    TB_DIR = os.path.join(ROOT, "runs", "tb")
    LATEST_JSON = os.path.join(CKPT_DIR, "latest.json")
    GAME_DIR = os.path.join(ROOT, "env")

    @staticmethod
    def device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
