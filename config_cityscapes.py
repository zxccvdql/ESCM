'''
Configuration File Used for Cityscapes Training & Evaluation
'''
class Config:
    DATA_ROOT = "path/to/cityscapes/dataset"
    BACKBONE_NAME = 'ResNet34'
    CROP_H = 224
    CROP_W = 224
    # TASKS = ["depth"]
    TASKS = ["seg", "depth"]
    # TASKS_NUM_CLASS = [1]
    TASKS_NUM_CLASS = [19, 1]

    LAMBDAS = [1, 20]
    NUM_GPUS = 1
    BATCH_SIZE = 64 * NUM_GPUS
    MAX_ITERS = 20000 / NUM_GPUS
    DECAY_LR_FREQ = 4000 / NUM_GPUS
    DECAY_LR_RATE = 0.7
        
    INIT_LR = 1e-4
    WEIGHT_DECAY = 5e-4
    IMAGE_SHAPE = (256, 512)

    PRUNE_TIMES = 11
    PRUNE_ITERS = [100] * PRUNE_TIMES

    END = 15000 / NUM_GPUS
    INT = 50
    PRUNE_RATE = 0.5
    RETRAIN_EPOCH = 1000
    RETRAIN_DECAY_LR_FREQ = 1000 / NUM_GPUS
    RETRAIN_LR = 1e-5