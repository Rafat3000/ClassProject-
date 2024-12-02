# Machine Learning Model Training and Experimentation with MLflow  

This repository contains a Python script for training and evaluating various machine learning models using **MLflow** for tracking experiments. The script leverages models from **scikit-learn** and processes a dataset to predict target classes.  

## **Features**  
- Model tracking and logging with MLflow.  
- Support for multiple classifiers:  
  - K-Nearest Neighbors (KNN)  
  - Decision Tree  
  - Random Forest  
  - AdaBoost  
- Automatic feature scaling and label encoding for categorical data.  
- Logging of model parameters, accuracy, and experiment metadata.  

---

## **Requirements**  
Ensure you have the following packages installed:  

```bash  
pip install mlflow scikit-learn pandas numpy openpyxl
 ## Usage

1. Prepare Dataset
Place your dataset (glass (Imbalanced).xlsx) in the root directory. The dataset should have a column named Class as the target variable.

2. Run the Script
To run the script and log experiments:
python main.py  

3. View MLflow Dashboard
To track experiments and view metrics:
mlflow ui  
Open http://localhost:5000 in your browser to explore logged experiments.

---------------


# Machine Learning Model Training and Experimentation with MLflow  

This repository contains a Python script for training and evaluating various machine learning models using **MLflow** for tracking experiments. The script leverages models from **scikit-learn** and processes a dataset to predict target classes.  

## **Features**  
- Model tracking and logging with MLflow.  
- Support for multiple classifiers:  
  - K-Nearest Neighbors (KNN)  
  - Decision Tree  
  - Random Forest  
  - AdaBoost  
- Automatic feature scaling and label encoding for categorical data.  
- Logging of model parameters, accuracy, and experiment metadata.  

---

## **Requirements**  
Ensure you have the following packages installed:  

```bash  
pip install mlflow scikit-learn pandas numpy openpyxl
Usage

1. Prepare Dataset
Place your dataset (glass (Imbalanced).xlsx) in the root directory. The dataset should have a column named Class as the target variable.

2. Run the Script
To run the script and log experiments:

python main.py  
3. View MLflow Dashboard
To track experiments and view metrics:

mlflow ui  
Open http://localhost:5000 in your browser to explore logged experiments.

## Script Overview

Data Processing:
  - The script reads data from glass (Imbalanced).xlsx.
  - It handles categorical encoding and scales features using MinMaxScaler.
Model Training:
Trains models using four different classifiers:
  - KNeighborsClassifier
  - DecisionTreeClassifier
  - RandomForestClassifier
  - AdaBoostClassifier
MLflow Logging:
  - Logs parameters, accuracy, and other metrics.

----------------
 ## Model Configuration

Modify the models list in the script to add or update models and their hyperparameters:
models = [  
    ("KNeighborsClassifier", KNeighborsClassifier(), {"n_neighbors": 3}),  
    ("DecisionTreeClassifier", DecisionTreeClassifier(), {}),  
    ("RandomForestClassifier", RandomForestClassifier(), {"n_estimators": 27, "criterion": 'entropy'}),  
    ("AdaBoostClassifier", AdaBoostClassifier(algorithm="SAMME"), {"n_estimators": 50})  
]  
----------------

## Contributions

Contributions are welcome! Feel free to fork the repository and submit a pull request.

----------------

