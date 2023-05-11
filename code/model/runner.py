import argparse
import tensorflow as tf
from scripts.download_dataset import download_data_from_gcp
from model.logger_utils import ImageLabelingLogger, ModelLogger
import model.hyperparameters as hp
from model.dataset import SantaDataset
from model.models.model import SantaClassifier
import os
from shared.utils import find_best_accuracy_from_saved_models, create_all_folders, get_run_num, make_nested_folder, create_run_file_name
from model.lr_scheduler import LRScheduler
from model.models.transfer_model import TransferSantaClassifier
from shared.constants import *
import sys
from shared.logging import log_summary, log_hyperparameters, log_model, log_data_augmentation
from shared.gcp import upload_to_gcp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-r", "--run_number", type=int, help="The run number we should use", default=None, required=False)
    arg_parser.add_argument("-t", "--transfer_learning", help="Whether we are running the transfer learning model or not", action="store_true", default=False, required=False)
    arg_parser.add_argument("-g", "--gcp", help="Upload to GCP", action="store_true", default=False, required=False)
    return arg_parser.parse_args()

def get_best_model_file(folder, model_num):
    path = os.path.join(folder, f"run_{model_num}")
    if not os.path.isdir(path):
        print(f"model {model_num} doesn't exist")
        return None
    best_model_folder, best_acc = find_best_accuracy_from_saved_models(path)
    if best_model_folder == "":
        print("did not find any best model files")
        return None
    print(f"found model file {best_model_folder} with best accuracy: {best_acc}")
    return best_model_folder

def create_metrics():
    metrics = [
        tf.keras.metrics.BinaryAccuracy(), 
        tf.keras.metrics.TruePositives(), 
        tf.keras.metrics.TrueNegatives(), 
        tf.keras.metrics.FalseNegatives(), 
        tf.keras.metrics.FalsePositives()
    ]
    return metrics

def train(model, run_number, datasets):
    paths = list()
    logs_dir = make_nested_folder(DEFAULT_LOGS_DIR, run_number)
    paths.append(logs_dir)
    
    r = create_run_file_name(run_number)
    info_save_file = os.path.join(DEFAULT_INFO_DIR, f"{r}.txt")
    paths.append(info_save_file)
    
    saved_models_dir = make_nested_folder(DEFAULT_SAVED_MODELS_DIR, run_number)
    paths.append(saved_models_dir)
    
    lr_file = os.path.join(DEFAULT_LR_DIR, "f{r}.txt")
    paths.append(lr_file)
    
    print(f"using log dir: {os.path.basename(logs_dir)}")
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_dir,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_dir,  datasets),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=hp.patience,
            verbose=1,
            restore_best_weights=True
        ),
        LRScheduler(lr_file),
        ModelLogger(info_save_file, saved_models_dir)
    ]
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.validation_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
        initial_epoch=0,
        class_weight=datasets.class_weights
    )
    return paths   
    
def test(model, run_num, data): 
    run_file_txt = f"{create_run_file_name(run_num)}.txt"
    output_file = os.path.join(DEFAULT_TEST_RESULTS_DIR, run_file_txt)
    paths = [output_file]
    with open(output_file, "w") as f:
        out = sys.stdout
        sys.stdout = f
        model.evaluate(
            x=data,
            verbose=1
        )
        sys.stdout = out
    return paths

def create_model(is_transfer_model, model_num, saved_model_dir):
    if model_num is not None:
        loaded_model_file = get_best_model_file(saved_model_dir, model_num)
        if loaded_model_file is None:
            return None
        return tf.keras.models.load_model(loaded_model_file)
    if is_transfer_model:
        model = TransferSantaClassifier()
    else:
        model = SantaClassifier()
    model(tf.keras.Input(shape=(hp.image_size, hp.image_size, 3)))
    model.summary()
    metrics = [
        tf.keras.metrics.BinaryAccuracy(), 
        tf.keras.metrics.TruePositives(), 
        tf.keras.metrics.TrueNegatives(), 
        tf.keras.metrics.FalseNegatives(), 
        tf.keras.metrics.FalsePositives()
    ]
    model.compile(
        optimizer=model.optimizer,
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=hp.gamma, apply_class_balancing=True),
        metrics=metrics)    
    return model
 
def log_model_details(run_number, model, is_transfer):
    run_file_txt = f"{create_run_file_name(run_number)}.txt"
    paths_list = list()
    summary_output_path = os.path.join(DEFAULT_SUMMARIES_DIR, run_file_txt)
    log_summary(summary_output_path, model)
    paths_list.append(summary_output_path)
    hp_output_path = os.path.join(DEFAULT_HP_DIR, run_file_txt)
    log_hyperparameters(hp_output_path)
    paths_list.append(hp_output_path)
    da_output_path = os.path.join(DEFAULT_DA_DIR, run_file_txt)
    log_data_augmentation(da_output_path)
    paths_list.append(da_output_path)
    if is_transfer:
        model_output_file = os.path.join(DEFAULT_TRANSFER_MODEL_LOGS_DIR, run_file_txt)
        log_model(model_output_file, True)
    else:
        model_output_file = os.path.join(DEFAULT_MODEL_LOGS_DIR, run_file_txt)
        log_model(model_output_file, False)    
    paths_list.append(model_output_file)
    return paths_list
    
def test_and_train(run_number, is_transfer_learning, gcp):
    if run_number is None:
        run_number = get_run_num(DEFAULT_SAVED_MODELS_DIR)
        print(f"this is run {run_number}")
    else:
        print(f"Using run {run_number}")
    create_all_folders()
    if gcp:
        if not os.path.exists(DEFAULT_DATA_DIR):
            download_data_from_gcp()
    dataset = SantaDataset(DEFAULT_DATA_DIR)
    model = create_model(is_transfer_learning, False, DEFAULT_SAVED_MODELS_DIR)
    
    paths = log_model_details(run_number, model, is_transfer_learning)
    
    more_paths = train(model, run_number, dataset)
    
    paths.extend(more_paths)
    model = create_model(is_transfer_learning, run_number, DEFAULT_SAVED_MODELS_DIR)
    
    more_paths = test(model, run_number, dataset.test_data)
    paths.extend(more_paths)
    if gcp:
        upload_to_gcp(paths)
    
    
    
    