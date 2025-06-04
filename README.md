# Project 2: Feature Selection with Nearest Neighbor

CS205 - Artificial Intelligence

## Project Description
Feature selection is a crucial preprocessing step in machine learning that helps in improving the model's accuracy, reducing overfitting, and enhancing computational efficiency by selecting relevant features and discarding redundant or irrelevant ones. In this project, we evaluated the performance of two popular feature selection techniques forward selection and backward elimination using the Nearest Neighbor classification algorithm. The evaluation method employed was "leave-one-out" cross-validation, which systematically uses each instance of the dataset as a test case while using the remaining instances for training. We analyzed the effectiveness of these methods across three distinct datasets: [Large Data (35)](https://www.dropbox.com/scl/fo/ydr7o2fo4ljbv0l5mmo7x/AAHQO7k2yBpVejVcIjN4mQk?dl=0&e=1&preview=CS205_large_Data__35.txt&rlkey=tikonndbdlen603v3ln51hsa1), [Small Data (35)](https://www.dropbox.com/scl/fo/ydr7o2fo4ljbv0l5mmo7x/AAHQO7k2yBpVejVcIjN4mQk?dl=0&e=1&preview=CS205_small_Data__35.txt&rlkey=tikonndbdlen603v3ln51hsa1), and the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality).

## How to Run the Code
1. Clone or download the repository containing the project files.
2. Open a terminal or command prompt and navigate to the directory containing the files.
3. Install the required dependencies using the following command:
   ```
   pip install -r requirements.txt
   ```
4. Run the `main.py` file using the following command:
   ```
   python main.py
   ```
5. Follow the prompts
    1. **Select the data source:**
    - Enter `1` to load a dataset from a local directory.
    - Enter `2` to fetch a dataset from the UC Irvine Dataset Repository.

    2. **If loading from a local directory:**
    - Input the file path of the dataset when prompted.

    3. **If fetching from the UC Irvine Dataset Repository:**
    - The script will automatically fetch the Wine Quality Dataset (ID: 186).

    4. **Select the algorithm to run:**
    - Enter `1` for Forward Selection.
    - Enter `2` for Backward Elimination.

    5. **Output:**
    - The script will display the accuracy of the nearest neighbor algorithm using all features.
    - It will then run the selected feature selection algorithm and display the best feature subset along with its accuracy.
    - The results will also be saved to `logs/feature_accuracy_output.txt`.

    6. **Logs:**
    - All outputs will be logged in `logs/logs.txt`.

## Output 

The program will display the following:

- The dataset details (number of features and instances).
- The accuracy of the nearest neighbor algorithm using all features.
- The search process for the selected feature selection algorithm, including:
  - The feature subsets being evaluated at each levels.
  - Their corresponding accuracies.
  - The best feature subset and its accuracy at each level.
- The total time taken for the algorithm.

### Example
Please check `logs/forward_small_logs.txt` and `logs/forward_small_accuracy_output.txt` file.


## Files in the Repository

- `main.py`: The main program file that implements feature selection algorithms (Forward Selection and Backward Elimination) for datasets. It includes functionality for loading datasets, normalizing features, and evaluating feature subsets using leave-one-out accuracy with a nearest neighbor classifier.

- `visualization.ipynb`: A Jupyter notebook that evaluates and visualizes the performance of the feature selection algorithms. It collects metrics such as accuracy, selected feature subsets, and generates plots.


Happy experimenting!
