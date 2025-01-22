COGS_LIST = ['general', 'database', 'elfvision']

CACHE_DIR = "vision/cache"
IMAGE_SIZE = 224

# Database
DATABASE_DIR = "database"
DATABASE_FILENAME = "elf_database.db"

# AI Training
INPUT_SHAPE = 3
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.01
MODEL_DIR = "vision/model"
MODEL_NAME = "ELFVision2.safetensors"
TRAIN_DIR = "vision/data/train"
TEST_DIR = "vision/data/test"