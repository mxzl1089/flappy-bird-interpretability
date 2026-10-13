import os
import torch

class Config:

    NUM_ACTIONS = 2
    FRAME_SIZE = 80
    STACK_LEN = 4

    GAMMA = 0.99
    GAE_LAMBDA = 0.95

    N_STEPS = 1024

    N_EPOCHS = 4

    BATCH_SIZE = 256
    LR = 2.5e-4
    LR_DECAY = True

    CLIP_RANGE = 0.2

    ENT_COEF = 0.01

    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    NORMALIZE_ADV = True

    NORM_ADV_PER_MINIBATCH = True

    CLIP_VLOSS = True

    TARGET_KL = None

    MAX_STEPS = 2000000
    SAVE_EVERY = 10000
    LOG_EVERY_UPDATES = 1
    USE_TB = True

    ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(ROOT)
    CKPT_DIR = os.path.join(ROOT, "runs")
    TB_DIR = os.path.join(ROOT, "runs", "tb")
    LATEST_JSON = os.path.join(CKPT_DIR, "latest.json")
    GAME_DIR = os.path.join(PROJECT_ROOT, "env")

    @staticmethod
    def device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
