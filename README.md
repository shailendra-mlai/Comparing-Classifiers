# Comparing-Classifiers
Professional Certificate in Machine Learning and Artificial Intelligence. Practical Application Assignment 17.1: Comparing Classifiers

**Overview**

Goal of the assigment is to compare the performance of the classifiers encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. We will utilize a dataset related to marketing bank products over the telephone.

**Getting Started**

Our dataset comes from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/dataset/222/bank+marketing). The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns. We will make use of the article accompanying the dataset for more information on the data and features.

**Understanding the Data**

To gain a better understanding of the data, please read the information provided in the UCI link above, and examine the Materials and Methods section of the paper. How many marketing campaigns does this data represent?

The dataset collected is related to 17 campaigns that occurred between May 2008 and November 2010, corresponding to a total of 79354 contacts

**Read in the Data**

Use pandas to read in the dataset bank-additional-full.csv and assign to a meaningful variable name.

**Understanding the Features**

Examine the data description below, and determine if any of the features are missing values or need to be coerced to a different data type.

Converting the datatype to best possilble datatypes

**Understanding the Task**

After examining the description and data, your goal now is to clearly state the Business Objective of the task. State the objective below.

1. The business objective is to train a model with given dataset to predict the success of a contact, specifically whether the client subscribes to the deposit.
2. The selected ML model holds the potential to enhance campaign efficiency by pinpointing key characteristics influencing success.
3. By using this machine learning model, we can manage resources like human effort, phone calls, and time better. It will help us choose high-quality and cost-effective potential buyers.

**Engineering Features**

Now that you understand your business objective, we will build a basic model to get started. Before we can do this, we must work to encode the data. Using just the bank information features (columns 1 - 7), prepare the features and target column for modeling with appropriate encoding and transformations.

Create Categorical Variables for ["job","marital","education","default","housing","loan"]

**Train/Test Split**

With your data prepared, split it into a train and test set.

**A Baseline Model**

Before we build our first model, we want to establish a baseline. What is the baseline performance that our classifier should aim to beat?

Determine the baseline score for the classifier by using the `DummyClassifier` with the training data.

**A Simple Model**

Use Logistic Regression to build a basic model on your data.

**Score the Model**

What is the accuracy of your model?
Simple Logistic Regression accuracy is 0.887

**Model Comparisons**

Now, we aim to compare the performance of the Logistic Regression model to our KNN algorithm, Decision Tree, and SVM models. Using the default settings for each of the models, fit and score each. Also, be sure to compare the fit time of each of the models.

<img width="441" alt="Base_model" src="https://github.com/shailendra-mlai/Comparing-Classifiers/assets/153253910/da0dd510-cbf7-47f6-a2aa-5f8d1f058cd1">

**Improving the Model**

Now that we have some basic models on the board, we want to try to improve these. Below, we list a few things to explore in this pursuit.

More feature engineering and exploration.
Hyperparameter tuning and grid search. All of our models have additional hyperparameters to tune and explore. For example the number of neighbors in KNN or the maximum depth of a Decision Tree.

1. Ran LogisticRegression with max_iter = 1000
2. Ran KNeighborsClassifier with n_neighbors=18 got the best scores
3. Ran Decision Tree Classifier with max depth 1 to 10 and found max depth = 4, got best accuracy
4. Ran SVC with different Kernels ['rbf', 'poly', 'linear', 'sigmoid'] and found Kernel 'rbf with gamma = 0.1 achieved the best scores. 

Adjust your performance metric:

<img width="439" alt="improved_model" src="https://github.com/shailendra-mlai/Comparing-Classifiers/assets/153253910/e4e60d78-232d-4159-b39b-a07a66c491ad">


**Conclusion and Recommendations**

The business objective is to develop a machine learning model that accurately predicts whether a client will subscribe to a deposit. This model has the potential to enhance campaign efficiency by identifying key characteristics that influence success. By leveraging this model, we can improve resource management, optimize human effort, phone calls, and time allocation, and select a high-quality and cost-effective pool of prospective buyers. Overall, the implementation of this model holds great promise for achieving our business objectives and maximizing the effectiveness of our campaigns.

Model Selection: Among the classifiers tested (Logistic Regression, K Nearest Neighbors, Decision Tree, and Support Vector Machines), the best-performing model is the one with the highest accuracy and acceptable fit time. In this case, the Logistic Regression model with an adjusted max_iter of 1000 shows promising results, achieving an accuracy of 0.887.

Feature Engineering: Further exploration of feature engineering techniques could potentially improve model performance. This includes exploring additional features, transforming existing ones, or creating new ones that may better capture the underlying patterns in the data. Incorporate 13 additional features from the dataset. Select model with highest accuracy and precision.

Hyperparameter Tuning: Fine-tuning hyperparameters for each model can lead to improvements in performance. For instance, in K Nearest Neighbors, setting the number of neighbors to 18 resulted in the best scores, while in Decision Tree, a maximum depth of 4 yielded the highest accuracy. Similarly, for Support Vector Machines, the 'rbf' kernel with a gamma value of 0.1 achieved the best results.

Evaluation Metric: While accuracy is a commonly used metric, it's important to consider other evaluation metrics such as precision, recall, and F1-score, especially in scenarios where class imbalances exist or different costs are associated with false positives and false negatives.

By implementing these recommendations, we can further improve the performance of our models and make more accurate predictions regarding the success of a contact subscribing to the deposit, thereby enhancing campaign efficiency and resource management.







