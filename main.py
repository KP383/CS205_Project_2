import numpy as np
from itertools import combinations
from collections import defaultdict
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

def load_dataset(file_path):
    data = np.loadtxt(file_path)
    X = data[:, 1:]
    y = data[:, 0]
    return X, y

def normalize_features(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def one_nearest_neighbor(X_train, y_train, test_instance, feature_subset):
    distances = np.linalg.norm(X_train[:, feature_subset] - test_instance, axis=1)
    nearest_index = np.argmin(distances)
    return y_train[nearest_index]

def leave_one_out_accuracy(X, y, feature_subset):
    correct = 0
    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        test_instance = X[i, feature_subset]
        test_label = y[i]
        predicted = one_nearest_neighbor(X_train, y_train, test_instance, feature_subset)
        if predicted == test_label:
            correct += 1
    return (correct / len(X)) * 100

def exhaustive_forward_selection(X, y):
    n_features = X.shape[1]
    all_results = defaultdict(float)
    best_overall = []
    best_accuracy = 0.0

    print("\n>>> Forward Selection (Exhaustive Tree Search)")
    print("-" * 60)

    for level in range(1, n_features + 1):
        print(f"Level {level}:")
        best_this_level = None
        best_this_level_acc = 0.0

        for subset in combinations(range(n_features), level):
            acc = leave_one_out_accuracy(X, y, list(subset))
            
            print(f"  Trying feature(s): {list(subset)} -> Accuracy: {acc:.1f}%")

            if acc > best_this_level_acc:
                best_this_level = subset
                best_this_level_acc = acc
                

            if acc > best_accuracy:
                best_overall = list(subset)
                best_accuracy = acc

        if best_this_level:
            all_results[best_this_level] = best_this_level_acc
            print(f"-> Selected feature set at level {level}: {list(best_this_level)}, Accuracy: {best_this_level_acc:.1f}%")
        print(f"-> Best feature set so far: {best_overall}, Accuracy: {best_accuracy:.1f}%")
        print("-" * 60)

    print(f"\n>>> Finished Forward Selection! Best feature subset: {best_overall} with Accuracy: {best_accuracy:.1f}%\n")
    return best_overall, best_accuracy, all_results

def exhaustive_backward_elimination(X, y):
    n_features = X.shape[1]
    all_results = defaultdict(float)
    full_set = tuple(range(n_features))
    best_overall = list(full_set)
    best_accuracy = leave_one_out_accuracy(X, y, list(full_set))
    all_results[full_set] = best_accuracy

    print("\n>>> Backward Elimination (Exhaustive Tree Search)")
    print("-" * 60)
    print(f"Initial full feature set: {list(full_set)}, Accuracy: {best_accuracy:.1f}%\n")

    for level in range(n_features - 1, 0, -1):
        print(f"Level {n_features - level}:")
        best_this_level = None
        best_this_level_acc = 0.0

        for subset in combinations(range(n_features), level):
            acc = leave_one_out_accuracy(X, y, list(subset))
            print(f"  Trying feature(s): {list(subset)} -> Accuracy: {acc:.1f}%")

            if acc > best_this_level_acc:
                best_this_level = subset
                best_this_level_acc = acc

            if acc > best_accuracy:
                best_overall = list(subset)
                best_accuracy = acc

        if best_this_level:
            all_results[best_this_level] = best_this_level_acc
            print(f"-> Selected feature set at level {n_features - level}: {list(best_this_level)}, Accuracy: {best_this_level_acc:.1f}%")
        print(f"-> Best feature set so far: {best_overall}, Accuracy: {best_accuracy:.1f}%")
        print("-" * 60)

    print(f"\n>>> Finished Backward Elimination! Best feature subset: {best_overall} with Accuracy: {best_accuracy:.1f}%\n")
    return best_overall, best_accuracy, all_results

# -------- Main Execution --------
if __name__ == "__main__":

    sys.stdout = OutputLogger("output.txt")

    file_path = input("Enter dataset file path: ")
    X, y = load_dataset(file_path)
    X = normalize_features(X)
    print(f"This dataset has {len(X[0])} features (not including the class attribute), with {len(X)} instances.")
    print("\nSelect the algorithm to run:")
    print("1) Forward Selection")
    print("2) Backward Elimination")

    choice = int(input("Your choice: ").strip())
    start_time = time.time()
    if choice == 1:
        best_features, best_acc, result_dict = exhaustive_forward_selection(X, y)
    elif choice == 2:
        best_features, best_acc, result_dict = exhaustive_backward_elimination(X, y)
    else:
        print("Invalid choice.")
        exit()
    print(f"Total Time: {time.time() - start_time:.2f}/60 minutes")

    # Optional: save results for plotting
    with open("feature_accuracy_log.txt", "w") as f:
        for feat_set, acc in result_dict.items():
            f.write(f"{list(feat_set)}, {acc:.2f}\n")
