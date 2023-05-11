DEFAULT_INFO_DIR = "info"
DEFAULT_LR_DIR = "learning_rate"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_HP_DIR = "hyperparameters"
DEFAULT_DA_DIR = "data_augmentation"
DEFAULT_SAVED_MODELS_DIR = "saved_models"
DEFAULT_SUMMARIES_DIR = "summaries"
DEFAULT_CODED_MODELS_DIR = "code/model/models"
DEFAULT_DATA_DIR = "data"
DEFAULT_TEST_RESULTS_DIR = "test_results"
DEFAULT_MODEL_LOGS_DIR = "model_logs"
DEFAULT_TRANSFER_MODEL_LOGS_DIR = "transfer_model_logs"

# name of dir: does it correspond to a file
STATIC_CREATED_FOLDERS = {
    DEFAULT_INFO_DIR: True, 
    DEFAULT_LR_DIR: True ,
    DEFAULT_LOGS_DIR: False,
    DEFAULT_HP_DIR: True,
    DEFAULT_DA_DIR: True,
    DEFAULT_SAVED_MODELS_DIR: False,
    DEFAULT_SUMMARIES_DIR: True,
    DEFAULT_TEST_RESULTS_DIR: True,
    DEFAULT_MODEL_LOGS_DIR: True,
    DEFAULT_TRANSFER_MODEL_LOGS_DIR: True
}


# Data collection related constants
THRESHOLD = 10
IMAGE_SIZE = 256
NUM_IMAGES = 10000