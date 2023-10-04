# CSC219 PROJECT: 1 Heart Disease Detection using Neural Networks

Welcome to our team's machine learning project aimed at detecting heart disease. In this collaborative effort, we have tackled the crucial issue of heart disease detection using tabular data, which includes both categorical and numeric features. We've formulated this problem as a binary classification task, with the goal of predicting the presence (1) or absence (0) of heart disease. Throughout this project, we've explored various machine learning algorithms and rigorously evaluated their performance using essential metrics such as precision, recall, F1-score, along with generating insightful visualizations like confusion matrices and ROC curves.

## Dataset
We used a comprehensive heart disease dataset obtained from IEEE DataPort. This dataset contains information from multiple sources, including 11 common features relevant to heart disease diagnosis. Understanding the dataset's documentation was crucial to our work.

## Data Preprocessing
The data preprocessing steps included:

## Train-test split.
Handling missing values.
Removing duplicates.
Encoding categorical features.
Normalizing numeric features.
Machine Learning Models

## We explored three machine learning models for heart disease detection:

### Nearest Neighbor
We implemented a k-NN (k-Nearest Neighbors) classifier using scikit-learn and evaluated its performance.

### Support Vector Machine (SVM)
We built an SVM (Support Vector Machine) classifier using scikit-learn and assessed its classification performance.

### Fully-Connected Neural Networks
We created a fully connected neural network using TensorFlow and Keras. We used Bayesian optimization to fine-tune hyperparameters, including activation functions, neuron counts, and optimizers.

### Hyperparameter Tuning
We employed KerasTuner to find the optimal combination of hyperparameters for the fully connected neural network model. Hyperparameters included activation functions (e.g., relu, tanh), neuron counts, and optimizers (e.g., adam, sgd).

## Additional Features: We explored the following additional features:

### Feature Importance Analysis: 
Identifying the most important features using logistic regression and training models with only the top features.

### K-Means Clustering: 
Applying K-means clustering to the training set and using centroids to classify records in the test set.

### Balanced Dataset: 
Creating a more balanced dataset using oversampling or undersampling techniques to mitigate class imbalance.
