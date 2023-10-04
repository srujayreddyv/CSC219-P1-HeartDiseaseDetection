# CSC219 PROJECT: 1 Heart Disease Detection using Neural Networks
Welcome to our collaborative machine learning project aimed at detecting heart disease. In this project, we have addressed the critical issue of heart disease detection using tabular data, encompassing both categorical and numeric features. Our primary objective was to solve this problem as a binary classification task, distinguishing between the presence (1) or absence (0) of heart disease. Throughout our endeavor, we have explored various machine learning algorithms and conducted thorough evaluations based on essential metrics like precision, recall, F1-score. Additionally, we have provided insightful visualizations, including confusion matrices and ROC curves, to assess model performance.

## Dataset
To facilitate our heart disease detection task, we utilized a comprehensive heart disease dataset obtained from IEEE DataPort. This dataset amalgamates information from multiple sources and encompasses 11 common features relevant to heart disease diagnosis. Our work heavily relied on a profound understanding of the dataset's documentation.

## Data Preprocessing: 
Our data preprocessing pipeline comprised several essential steps:

1. Handling missing values: Any missing data points were appropriately handled to ensure data integrity.
2. Removing duplicates: Duplicate entries were identified and removed to maintain data consistency.
3. Encoding categorical features: Categorical features were encoded to make them compatible with machine learning algorithms.
4. Normalizing numeric features: Numeric features were normalized to ensure they were on the same scale, preventing any particular feature from dominating the learning process.
5. Train-test split: We partitioned the dataset into training and testing subsets.

## Building Models: 
We explored three machine learning models for heart disOur project ventured into the realm of machine learning by exploring three distinct models for heart disease detection:
1. Nearest Neighbor: We implemented a k-NN (k-Nearest Neighbors) classifier using scikit-learn and meticulously evaluated its performance.
2. Support Vector Machine (SVM): Leveraging scikit-learn, we constructed an SVM (Support Vector Machine) classifier and conducted an in-depth assessment of its classification performance.
3. Fully-Connected Neural Networks: To harness the power of deep learning, we designed and trained a fully connected neural network using TensorFlow and Keras. Our model tuning process involved Bayesian optimization, which allowed us to optimize hyperparameters, including activation functions, neuron counts, and optimizers.

## Hyperparameter Tuning
Hyperparameter tuning played a pivotal role in enhancing our model's performance. We utilized KerasTuner to systematically explore and identify the optimal combination of hyperparameters for our fully connected neural network model. These hyperparameters included activation functions such as relu and tanh, neuron counts, and optimizers like adam and sgd.

## Additional Features
In our quest to further improve our heart disease detection models, we explored the following additional features:

1. Feature Importance Analysis: We conducted a feature importance analysis, utilizing logistic regression to identify the most influential features. Subsequently, we trained models exclusively on these top features.
2. K-Means Clustering: We applied K-means clustering to the training dataset, leveraging the centroids to classify records in the test set.
3. Balanced Dataset: Recognizing the potential bias stemming from class imbalance, we implemented oversampling and undersampling techniques to create a more balanced dataset, mitigating this issue.
