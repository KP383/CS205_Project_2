import numpy as np
from collections import defaultdict
from ucimlrepo import fetch_ucirepo
import time
import sys

# Class to log output to both the terminal and a file
class OutputLogger:
    # Initialize the OutputLogger
    def __init__(self, file_path):
        # file_path: The name of the file to log output
        self.terminal = sys.stdout
        self.log = open(file_path, "w")

    # Write a message to both the terminal and the log file
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    # Flush the output streams
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Load the dataset from a file
def load_dataset(file_path):
    data = np.loadtxt(file_path)
    X = data[:, 1:]
    y = data[:, 0]
    return X, y

# Normalize the features (Z-score normalization)
def normalize_features(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# One nearest neighbor classification
def one_nearest_neighbor(X_train, y_train, test_instance, feature_subset):
    distances = np.linalg.norm(X_train[:, feature_subset] - test_instance, axis=1) # Calculate the distance between the test instance and the training instances
    nearest_index = np.argmin(distances) # Find the index of the nearest neighbor
    return y_train[nearest_index]

# Leave-one-out accuracy
def leave_one_out_accuracy(X, y, feature_subset):
    correct = 0
    for i in range(len(X)):
        # Delete the i-th instance from the training set
        X_train = np.delete(X, i, axis=0) 
        y_train = np.delete(y, i, axis=0) 
        test_instance = X[i, feature_subset] # Get the test instance
        test_label = y[i] # Get the test label
        # Predict the label for the test instance
        predicted = one_nearest_neighbor(X_train, y_train, test_instance, feature_subset) 
        if predicted == test_label:
            correct += 1
    return (correct / len(X)) * 100

# Forward selection
def forward_selection(X, y):
    n_features = X.shape[1] 
    selected = []
    all_results = defaultdict(float)
    best_overall = []
    best_accuracy = 0.0

    print("\n>>> Forward Selection (Greedy Search)")
    print("-" * 60)

    # Iterate through each level of the feature selection
    for level in range(1, n_features + 1):
        print(f"Level {level}:")
        best_this_level = None
        best_this_level_acc = 0.0

        for f in range(n_features): # Iterate through each feature
            if f in selected:
                continue
            trial = selected + [f] 
            acc = leave_one_out_accuracy(X, y, trial) # Calculate the accuracy of the trial
            print(f"  Trying features: {trial} -> Accuracy: {acc:.1f}%")
            if acc > best_this_level_acc: # If the accuracy of the trial is better than the best accuracy of the current level
                best_this_level = f 
                best_this_level_acc = acc

            if acc > best_accuracy: # If the accuracy of the trial is better than the best accuracy of the overall
                best_overall = trial[:]
                best_accuracy = acc 

        if best_this_level is not None:
            selected.append(best_this_level)
        all_results[tuple(selected)] = best_this_level_acc
    
        print(f"-> Selected feature set at level {level}: {set(selected)}, Accuracy: {best_this_level_acc:.1f}%")
        print(f"-> Best feature set so far: {set(best_overall)}, Accuracy: {best_accuracy:.1f}%")
        print("-" * 60)

    print(f"\n>>> Finished Forward Selection! Best feature subset: {best_overall} with Accuracy: {best_accuracy:.1f}%\n")
    return best_overall, best_accuracy, all_results

# Backward elimination
def backward_elimination(X, y):
    n_features = X.shape[1]
    selected = list(range(n_features))
    all_results = defaultdict(float)
    best_overall = selected[:]
    best_accuracy = leave_one_out_accuracy(X, y, selected)
    all_results[tuple(selected)] = best_accuracy

    print("\n>>> Backward Elimination (Greedy Search)")
    print("-" * 60)
    print(f"Initial full feature set: {selected}, Accuracy: {best_accuracy:.1f}%\n")

    # Iterate through each level of the feature selection
    for level in range(1, n_features):
        print(f"Level {level}:")
        best_this_level = None
        best_this_level_acc = 0.0

        for f in selected: # Iterate through each feature
            trial = selected[:] 
            trial.remove(f) 
            acc = leave_one_out_accuracy(X, y, trial) # Calculate the accuracy of the trial
            print(f"  Trying features: {trial} -> Accuracy: {acc:.1f}%")

            if acc > best_this_level_acc: # If the accuracy of the trial is better than the best accuracy of the current level
                best_this_level = f
                best_this_level_acc = acc

            if acc > best_accuracy: # If the accuracy of the trial is better than the best accuracy of the overall
                best_overall = trial[:] 
                best_accuracy = acc 

        if best_this_level is not None:
            selected.remove(best_this_level)
        
        all_results[tuple(selected)] = best_this_level_acc

        print(f"-> Selected feature set at level {level}: {set(selected)}, Accuracy: {best_this_level_acc:.1f}%")
        print(f"-> Best feature set so far: {set(best_overall)}, Accuracy: {best_accuracy:.1f}%")
        print("-" * 60)

    print(f"\n>>> Finished Backward Elimination! Best feature subset: {best_overall} with Accuracy: {best_accuracy:.1f}%\n")
    return best_overall, best_accuracy, all_results

# Main Execution
if __name__ == "__main__":

    sys.stdout = OutputLogger("logs/logs.txt")

    print("\nSelect the data source:")
    print("1) Local Directory")
    print("2) UC Irvine Dataset Repository")

    data_source_choice = int(input("Your choice: ").strip())

    if data_source_choice == 1:
        file_path = input("Enter dataset file path: ")
        X, y = load_dataset(file_path)
        print(f"This dataset has {len(X[0])} features (not including the class attribute), with {len(X)} instances.")
    
    elif data_source_choice == 2:
        dataset_id = 186 # Wine Quality Dataset ID
        dataset = fetch_ucirepo(id=dataset_id) # Fetch the dataset from the UC Irvine Dataset Repository
        X = dataset.data.features # Get the features of the dataset
        y = dataset.data.targets # Get the targets of the dataset
        X = X.to_numpy() # Convert the features to a numpy array
        y = y.to_numpy() # Convert the targets to a numpy array
        print(f"Dataset from UC Irvine Dataset Repository(ID: {dataset_id} - Wine Quality)")
        print(f"This dataset has {X.shape[1]} features (not including the class attribute), with {X.shape[0]} instances.")
    
    else:
        print("Invalid choice.")
        exit()

    X = normalize_features(X) # Normalize the features
    print("\nSelect the algorithm to run:")
    print("1) Forward Selection")
    print("2) Backward Elimination")

    choice = int(input("Your choice: ").strip())

    accuracy = leave_one_out_accuracy(X, y, list(range(X.shape[1]))) # Calculate the accuracy of the model with all features
    print(f"\nRunning Nearest Neighbor with all {X.shape[1]} features, using 'leave-one-out' evalution, I get an accuracy of {accuracy:.2f}%")
    
    print("-" * 60)
    start_time = time.time() 
    if choice == 1: # Forward Selection
        best_features, best_acc, result_dict = forward_selection(X, y)
    elif choice == 2: # Backward Elimination
        best_features, best_acc, result_dict = backward_elimination(X, y)
    else:
        print("Invalid choice.")
        exit()
    print(f"Total Time: {(time.time() - start_time):.2f} seconds") # Print the total time taken to run the algorithm

    # Write the results to a file
    with open("logs/feature_accuracy_output.txt", "w") as f:
        for feat_set, acc in result_dict.items():
            f.write(f"{list(feat_set)}, {acc:.2f}\n")
