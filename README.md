# THE SYRIATEL CUSTOMER CHURN ANALYSIS
## 1.0 Project Overview
This project focuses on building a predictive model to identify customers at high risk of churning from Syriatel, a telecommunications company. Customer churn is a situation where customers stop doing business with a company. In the highly competitive telecom sector, acquiring a new customer is far more expensive than retaining an existing one.

The primary goal is to analyze historical customer data to uncover the key drivers of churn and develop a machine learning model that can accurately predict which customers are likely to leave. The insights derived from this analysis will empower Syriatel to implement proactive retention strategies to improve customer loyalty and reduce revenue loss.

## 2.0 Business and Data Understanding 
Stakeholder Audience

The primary stakeholders for this project are:

1. Syriatel Management & Marketing Teams: They need to understand why customers churn to formulate effective business strategies and allocate retention resources efficiently.

2. Data Science Team: To understand the methodology, features used, and model performance for future improvements.

Dataset Choice: Syriatel Customer Churn Dataset

This project utilizes a publicly available [Syriatel Telecom Customer Churn Dataset](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset) from Kaggle containing information about Syriatel's customers. The dataset contains 3333 customer records and 21 features therefore, it is ideal for this task because it contains a rich mix of demographic, account, and service usage information that is typical in the telecom industry and highly relevant for predicting churn.

Key Features:

The features can be categorized into several groups that are critical for understanding customer behavior:

Target Variable:

`churn` - A binary indicator (True/False) of whether the customer left the service.

Customer Demographics and Account Info:

`state`: The  state of the customer (categorical).

`account_length`: The number of days the customer has had an account (numerical).

`area_code`: The area code of the customer (categorical).

Service Subscriptions:

`international_plan`: Whether the customer has an international plan (Yes/No).

`voice_mail_plan`: Whether the customer has a voice mail plan (Yes/No).

Customer Usage Patterns (Numerical):

`total_day_charge`: The total charges incurred by the customer for daytime calls. 

`total_eve_charge`: The total charges for calls made during the evening. 

`total_night_charge`: The total charges for nighttime calls. 

`total_intl_charg`e: The total charges for international calls.

Customer Service Interaction:

`customer_service_calls`: The number of times the customer has called service support (numerical).

## 3.0 Data Cleaning 
Validation & Initial Inspection: The first step was a thorough inspection of the dataset. We confirmed the dataset contained 3,333 entries and 21 columns. We checked for correct data types and ensured all values fell within expected ranges (e.g., non-negative call counts and charges), finding no anomalies.

Handling Missing Values: The dataset was verified to have no missing values, which simplified the cleaning process and ensured no imputation strategies were needed.

Handling Duplicates: The dataset was checked for completely duplicate customer records. No duplicates were found, ensuring each row represented a unique customer.

Consistency & Uniformity (Encoding): Categorical string values were standardized into a numerical format to make them suitable for machine learning algorithms.

Feature Engineering & Selection:

The phone number column was identified as a unique identifier and not a predictive feature; it was dropped to prevent model overfitting.

To avoid multicollinearity (which can destabilize models), the highly correlated minutes columns (e.g., total day minutes) were removed. The corresponding charge columns were retained as they directly represent customer spend and revenue.

Outlier Treatment: Numerical features (e.g., customer service calls, charge columns) were reviewed for extreme values. These were analyzed and mostly retained as they were deemed to represent genuine, albeit unusual, customer behavior rather than data entry errors. This ensures the model learns from the full spectrum of customer activity.

Feature Scaling: Finally, all numerical features were standardized using StandardScaler. This process transforms the data to have a mean of 0 and a standard deviation of 1, which is a critical step for models that rely on distance calculations and improves the convergence speed for many other algorithms.

## 4.0 Modeling approach
The problem is framed as a supervised binary classification task. The goal is to predict a binary outcome: churn (True/1) or not churn (False/0).

Models Trained and Evaluated:
Several machine learning algorithms were selected for their suitability to classification problems:

Logistic Regression: A strong, interpretable baseline model.

Random Forest Classifier: A powerful ensemble method that combines multiple decision trees to improve predictive accuracy and control overfitting.

Decision Tree Classifier: A non-parametric model used to understand the underlying patterns and serve as a benchmark for the Random Forest.

K-Nearest Neighbors (KNN): A simple, instance-based learning algorithm.

Methodology:

The preprocessed dataset was split into training and testing sets.

Models were trained on the training data (X_train_final, y_train_final).

Each trained model was used to make predictions on the held-out test set (X_test_full).

Predictions were evaluated against the true test values (y_test) using a comprehensive set of metrics.

## 5.0 Evaluation
In this stage, we assessed how well our models performed and determined whether they met the business objectives of predicting and reducing customer churn.

Model Comparison – Key Metrics

We compared Logistic Regression, Random Forest, Decision Tree, and K-Nearest Neighbors (KNN) using accuracy, precision, recall, and F1-score.

Random Forest achieved the highest performance with 91.9% accuracy and an F1-score of 70%, making it the most reliable model for churn prediction.

Decision Tree performed moderately with balanced recall and accuracy.

Logistic Regression and KNN achieved similar results but struggled with precision, making them less suitable for deployment.

Model Comparison – ROC Curves

The ROC curves and AUC scores further highlighted Random Forest as the best-performing model. Its curve stayed closest to the top-left corner of the plot, indicating strong discriminatory power between churners and non-churners.

Hyperparameter Tuning – Random Forest

We optimized the Random Forest model using GridSearchCV with cross-validation. The best parameters included:

n_estimators = 200

max_depth = 10

min_samples_split = 5

class_weight = balanced

After tuning:

The model achieved 91% accuracy on the test set.

It correctly identified 71% of churners (recall) while maintaining 69% precision, balancing the trade-off between false positives and false negatives.

Feature importance analysis revealed that customer service calls, international plans, and day-time charges were the strongest predictors of churn.

## 6.0 Conclusion
The analysis of the Syriatel customer churn dataset provided valuable insights into the factors that drive customer cjurn. Through exploratory analysis and predictive modeling, we identified customer service calls, international plans, and day-time charges as the strongest indicators of churn.

Among the models tested, the Random Forest Classifier consistently outperformed others, achieving 91% accuracy and correctly identifying 71% of churners after hyperparameter tuning. This balance between accuracy and recall makes it a practical tool for churn prediction in real-world scenarios.

Key Business Insights:

Customers with frequent customer service interactions are more likely to leave, highlighting a need for improved customer support.

The international calling plan is strongly associated with churn, suggesting pricing or satisfaction issues.

High day-time usage and charges increase the likelihood of churn, indicating potential dissatisfaction with costs.

Recommendations:

Enhance customer support services to reduce complaints and frustrations.

Reassess pricing strategies for international and high-usage plans.

Implement targeted retention campaigns focused on customers flagged as high risk by the churn model.

Continuously monitor churn metrics and update the predictive model with new customer data to maintain accuracy.

Limitations & Future Work

Data imbalance: Despite using SMOTE and class weights, the dataset still had fewer churn cases than non-churn, which can limit model generalization.

Feature limitations: The dataset did not include customer demographics (e.g., age, income, contract type) that could provide a deeper understanding of churn behavior.

Business context: The dataset is static and may not reflect current customer dynamics or external factors such as competitor actions.

Model deployment: The project remains at the analysis stage; deploying the model in a real-time environment would be the next practical step.

Next Steps:

Deploy the Random Forest model into Syriatel’s CRM system for real-time churn prediction.

Use model insights to personalize offers and promotions.

Track the effectiveness of retention strategies and refine them based on customer feedback and model updates.