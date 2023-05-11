import os
import re
import argparse
import matplotlib.pyplot as plt
from shared.utils import calculate_precision_recall_f1
from shared.constants import DEFAULT_SUMMARIES_DIR, DEFAULT_TEST_RESULTS_DIR, DEFAULT_LR_DIR, DEFAULT_INFO_DIR

def parse_args():
    parser = argparse.ArgumentParser(description="Create graphs for the specified metric.")
    parser.add_argument("-r", "--run_number", type=int, help="The run number for the data.", required = True)
    parser.add_argument("-c", "--combine", action="store_true", help="Combine training and validation metrics on the same graph.", default=False)
    parser.add_argument("-t", "--test", action="store_true", help="enable on test data.", default=False)
    return parser.parse_args()

def parse_num_model_params(run_number):
    file_path = os.path.join(DEFAULT_SUMMARIES_DIR, f"run_{run_number}.txt")    
    if not os.path.exists(file_path):
        return None
    total_params, trainable_params, non_trainable_params = None, None, None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Total params:"):
                total_params = int(line.split(":")[1].strip().replace(',', ''))
            elif line.startswith("Trainable params:"):
                trainable_params = int(line.split(":")[1].strip().replace(',', ''))
            elif line.startswith("Non-trainable params:"):
                non_trainable_params = int(line.split(":")[1].strip().replace(',', ''))
    return total_params, trainable_params, non_trainable_params

def parse_testing_data(file_path):
    
    if not os.path.exists(file_path):
        return None
    file_data = dict()
    file_data["loss"] = list()
    file_data["binary_accuracy"] = list()
    file_data["true_positives"] = list()
    file_data["true_negatives"] = list()
    file_data["false_positives"] = list()
    file_data["false_negatives"] = list()
    file_data["precision"] = list()
    file_data["recall"] = list()
    file_data["f1_score"] = list()
    
    with open(file_path, 'r') as file:
        pattern = r"loss: (\d+\.\d+) - binary_accuracy: (\d+\.\d+) - true_positives: (\d+\.\d+e\+\d+) - true_negatives: (\d+\.\d+) - false_negatives: (\d+\.\d+e\+\d+) - false_positives: (\d+\.\d+)"
        for line in file:
            matches = re.findall(pattern, line)
            for match in matches:
                loss, acc, tp, tn, fn, fp = float(match[0]), float(match[1]), float(match[2]), float(match[3]), float(match[4]), float(match[5])
                file_data["loss"].append(loss)
                file_data["binary_accuracy"].append(acc)
                file_data["true_positives"].append(tp)
                file_data["true_negatives"].append(tn)
                file_data["false_negatives"].append(fn)
                file_data["false_positives"].append(fp)
                prec, recall, f1 = calculate_precision_recall_f1(tp, fp, fn)
                file_data["precision"].append(prec)
                file_data["recall"].append(recall)
                file_data["f1_score"].append(f1)
                
    return {"Testing": file_data}
        
def parse_learning_rate(run_number, num_decimals=4):
    file_path = os.path.join(DEFAULT_LR_DIR, f"run_{run_number}.txt")
    if not os.path.exists(file_path):
        return None
    learning_rates = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("\tLearning Rate:"):
                lr = float(line.split(":")[1].strip())
                lr_truncated = round(lr, num_decimals)
                learning_rates.append(lr_truncated)
    return learning_rates

def parse_training_and_validation_data(run_number):
    file_path = os.path.join(DEFAULT_INFO_DIR, f"run_{run_number}.txt")
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as file:
        file_data = {'Training': {}, 'Validation': {}}
        epoch = -1
        section = None
        for line in file:
            line = line.strip()
            if line.startswith('Epoch:'):
                epoch = int(re.findall(r'\d+', line)[0])
            elif line.startswith('Training:') or line.startswith('Validation:'):
                section = line[:-1]
            elif line and epoch >= 0:
                key, value = line.split(':')
                formatted_key = key.strip().lower().replace(" ", "_")
                if formatted_key not in file_data[section]:
                    file_data[section][formatted_key] = list()
                file_data[section][formatted_key].append(float(value.strip()))
    return file_data

def create_graphs(run_number, data, metric, combine):
    save_path = os.path.join("graphs", f"run_{run_number}")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    sections = data.keys()
    overall_max, overall_min = float("-inf"), float("inf")
    for section in sections:
        if metric not in data[section]:
            continue
        epochs = [x for x in range(len(data[section][metric]))]
        values = [data[section][metric][epoch] for epoch in epochs]
        if not combine:
            plt.figure(figsize=(10, 5))
        plt.plot(epochs, values, label=f"{section} {metric}")
        overall_max = max(overall_max, max(values))
        overall_min = min(overall_min, min(values))
        if not combine:
            plt.xlim(min(epochs), max(epochs))
            plt.ylim(min(values), max(values))
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            if plt.gca().get_legend_handles_labels()[0]:
                plt.legend()
            plt.title(f"{section} {metric} Run {run_number}")
            plt.savefig(os.path.join(save_path, f"{section.lower()}_{metric}_run_{run_number}.png"))
            plt.close()
    if not combine:
        return
    plt.figure(figsize=(10, 5))
    plt.xlim(min(epochs), max(epochs))
    plt.ylim(overall_min, overall_max)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    if plt.gca().get_legend_handles_labels()[0]:
            plt.legend()
    plt.title(f"Combined {metric} Run {run_number}")
    plt.savefig(os.path.join(save_path, f"Combined_{metric}_run_{run_number}.png"))
    plt.close()

def choose_metric_to_use(metrics): 
    metrics = sorted(metrics)
    print("Available metics:")
    for i, metric in enumerate(metrics, 1):
        print(f"\t{i}. {metric}")
    metric_index = int(input("Enter the number of the metric you want to use: ")) - 1
    return metrics[metric_index]
 
def combine_graphs():
    user = input("do you want a combined train/vaidation: ") 
    user = user.lower().strip()
    if user == "y" or user == "yes":
        return True
    return False
    
    
def main():
    args = parse_args()
    run_number = args.run_number
    data = parse_training_and_validation_data(run_number)
    lr_data = parse_learning_rate(run_number)
    toal_params, trainable_params, non_trainable_params = parse_num_model_params(run_number)
    data["Training"]["learning_rate"] = lr_data
    if data is None:
        print("error parsing train/validation")
        return
    if not os.path.isdir("graphs"):
        os.makedirs("graphs")
    metrics = data["Training"].keys()
    chosen_metric = choose_metric_to_use(metrics)
    
    combine = args.combine

    create_graphs(args.run_number, data, chosen_metric, combine)    
    
    if args.test:
        file_path = os.path.join(DEFAULT_TEST_RESULTS_DIR, f"run_{run_number}.txt")
        test_data = parse_testing_data(file_path)
        if test_data is None:
            print("testing data could not be parsed... its possible the file doesn't exist")
            return
        metrics = test_data["Testing"].keys()
        chosen_metric = choose_metric_to_use(metrics)
        create_graphs(args.run_number, test_data, chosen_metric, False)
    