# Configuration of training
# Hyper params config (Using the best performance on page 11)
alp1 = 1
alp2 = 1
alp3 = 1

ld1 = 1
ld2 = 1

# LR params
mu1 = 2e-4  # For J loss
mu2 = 2e-4  # For L loss

BS = 64
feature_size = 200
EPCH = 10000

# Device selection
DEV = "cuda:1"

# Model export and load path
EXPORT_PATH = 'training/semi_supervised/exported/model_weight/'
LOG_PATH = 'training/semi_supervised/exported/'
