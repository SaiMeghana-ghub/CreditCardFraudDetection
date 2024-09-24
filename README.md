### FindDefault (Prediction of Credit Card fraud)--Capstone Project

# Problem Statement:

A credit card is one of the most used financial products to make online purchases and payments. Though the Credit cards can be a convenient way to manage your finances, they can also be risky. Credit card fraud is the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash.
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. 
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

# About Credit Card Fraud Detection:

Credit card fraud detection involves identifying unauthorized and fraudulent transactions to protect financial institutions and customers. The challenge lies in accurately predicting fraudulent activities within a highly imbalanced dataset, where genuine transactions vastly outnumber fraudulent ones. This project uses a dataset of European cardholders' transactions from September 2013, with only 492 frauds out of 284,807 transactions. The project encompasses data exploration, cleaning, balancing techniques, feature engineering, and the development of machine learning models. The ultimate goal is to deploy a robust system that can effectively identify and prevent fraudulent transactions in real-time.

# Project Introduction:

Credit card fraud, the unauthorized use of someone else's credit card for transactions, poses significant risks to financial institutions and customers. This project aims to develop a classification model to predict fraudulent transactions using a dataset of European cardholders' transactions from September 2013. With 492 frauds out of 284,807 transactions, the data is highly imbalanced. The project involves exploratory data analysis, data cleaning, handling imbalanced data, feature engineering, model selection, training, validation, and deployment, ultimately providing a robust solution to identify and prevent credit card fraud effectively.

# Project Outline:

   Exploratory Data Analysis: Analyze and understand the data to identify patterns, relationships, and trends in the data by using Descriptive Statistics and Visualizations.

	Data Cleaning: This might include standardization, handling the missing values and outliers in the data.

	Dealing with Imbalanced data: This data set is highly imbalanced. The data should be balanced using the appropriate methods before moving onto model building.

	Feature Engineering: Create new features or transform the existing features for better performance of the ML Models. 

	Model Selection: Choose the most appropriate model that can be used for this project. 

	Model Training: Split the data into train & test sets and use the train set to estimate the best model parameters. 

	Model Validation: Evaluate the performance of the model on data that was not used during the training process. The goal is to estimate the model's ability to generalize to new, unseen data and to identify any issues with the model, such as overfitting. 

	Model Deployment: Model deployment is the process of making a trained machine learning model available for use in a production environment. 


# Project Work Overview:
Our dataset exhibits significant class imbalance, with the majority of transactions being non-fraudulent (99.82%). This presents a challenge for predictive modeling, as algorithms may struggle to accurately detect fraudulent transactions amidst the overwhelming number of legitimate ones. To address this issue, we employed various techniques such as undersampling, oversampling, and synthetic data generation.

# Undersampling
We utilized the NearMiss technique to balance the class distribution by reducing the number of instances of non-fraudulent transactions to match that of fraudulent transactions. This approach helped in mitigating the effects of class imbalance. Our attempt to address class imbalance using the NearMiss technique did not yield satisfactory results. Despite its intention to balance the class distribution, the model's performance was suboptimal. This could be attributed to the loss of valuable information due to the drastic reduction in the majority class instances, leading to a less representative dataset. As a result, the model may have struggled to capture the intricacies of the underlying patterns in the data, ultimately affecting its ability to accurately classify fraudulent transactions.

# Oversampling:
To further augment the minority class, we applied the SMOTETomek method with a sampling strategy of 0.75. This resulted in a more balanced dataset, enabling the models to better capture the underlying patterns in fraudulent transactions.
# Machine Learning Models: 
After preprocessing and balancing the dataset, we trained several machine learning models, including:

Logistic Regression
K-Nearest Neighbors (KNN)
Random Forest Classifier
AdaBoost Classifier
XGBoost Classifier
# Evaluation Metrics:
We evaluated the performance of each model using various metrics such as accuracy, precision, recall, and F1-score. Additionally, we employed techniques like cross-validation and hyperparameter tuning to optimize the models' performance.

# Model Selection:
Among the various models and balancing methods experimented with, the XGBoost model stands out as the top performer when using oversampling techniques. Despite the inherent challenges posed by imbalanced datasets, the XGBoost algorithm demonstrates robustness and effectiveness in capturing the underlying patterns associated with fraudulent transactions. By generating synthetic instances of the minority class through oversampling methods like SMOTETomek, the XGBoost model achieves a more balanced representation of the data, enabling it to learn and generalize better to unseen instances. This superior performance underscores the importance of leveraging advanced ensemble techniques like XGBoost, particularly in the context of imbalanced datasets characteristic of credit card fraud detection.
# Future Work:
# Real-Time Fraud Detection:
Implementing a real-time fraud detection system that can analyze transactions as they occur, providing immediate alerts and preventing fraudulent transactions before they are completed.

# Advanced Machine Learning Models: 
Exploring and integrating more sophisticated machine learning and deep learning models, such as recurrent neural networks (RNNs) or transformers, which may capture complex temporal patterns and improve detection accuracy.

# Anomaly Detection Techniques: 
Incorporating unsupervised anomaly detection methods to identify new and evolving fraud patterns that were not present in the training data.
# Ensemble Methods: 
Utilizing ensemble learning techniques, such as stacking or boosting, to combine multiple models and improve overall performance and robustness.

# Feature Enrichment: 
Enhancing the feature set with additional data sources, such as device information, geolocation data, and user behavior patterns, to provide a more comprehensive view of each transaction.

# Continuous Model Training:
Establishing a continuous learning pipeline where the model is periodically retrained with the latest data to adapt to new fraud tactics and maintain high performance.

# Explainability and Interpretability: 
Developing explainable AI techniques to make the fraud detection model more transparent and understandable to stakeholders, ensuring trust and regulatory compliance.

# User Feedback Integration:
Incorporating user feedback mechanisms to refine and improve the model based on real-world usage and insights from fraud analysts.

# Scalability and Optimization:
Optimizing the deployment infrastructure to handle large volumes of transactions efficiently and ensure the system scales with increasing data and usage.

# Cross-Industry Collaboration:
Collaborating with other financial institutions and industry bodies to share knowledge, data, and strategies for more effective fraud detection across the sector.
# Project Assets Overview:
[data](https://github.com/SaiMeghana-ghub/CreditCardFraudDetection/blob/main/creditcard.csv):
This folder contains the dataset(s) used in the project. Includes raw.csv file along with a README file explaining the dataset's attributes. The 'preprocessed_data.csv' and 'resampled_os.csv' data files have been excluded from the repository by adding them to the .gitignore file due to their large file sizes.

[Notebooks](https://github.com/SaiMeghana-ghub/CreditCardFraudDetection/blob/main/creditcard-1.ipynb):
This folder contains Jupyter notebooks consisting all the project work i.e. data exploration, preprocessing, feature engineering, model building, model training and evaluation for the best fitting model along with detailed explaination.

models:
In this credit card fraud detection project, we utilized several machine learning models, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting Machines (GBM), and XGBoost. These models were chosen for their robustness, accuracy, and ability to handle the imbalanced nature of the dataset effectively.





