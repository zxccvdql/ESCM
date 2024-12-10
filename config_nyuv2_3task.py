'''
Configuration File Used for nyuv2 Training & Evaluation
'''
class Config:
    DATA_ROOT = "path/to/nyuv2/dataset"
    BACKBONE_NAME = 'ResNet34'
    CROP_H = 321
    CROP_W = 321
    TASKS = ["seg", 'sn', "depth"]
    TASKS_NUM_CLASS = [40, 3, 1]

    LAMBDAS = [1, 20, 3]
    NUM_GPUS = 1
    BATCH_SIZE = 16 * NUM_GPUS
    MAX_ITERS = 20000 / NUM_GPUS
    DECAY_LR_FREQ = 4000 / NUM_GPUS
    DECAY_LR_RATE = 0.7
        
    INIT_LR = 1e-4
    WEIGHT_DECAY = 1e-4
    IMAGE_SHAPE = (480, 640)

    PRUNE_TIMES = 11
    PRUNE_ITERS = [100] * PRUNE_TIMES

    # --------------------------------------------------------------- #
    END = 15000
    INT = 50
    PRUNE_RATE = 0.5
    RETRAIN_EPOCH = 1000 / NUM_GPUS
    RETRAIN_LR = 1e-5
    RETRAIN_DECAY_LR_FREQ = 1000
