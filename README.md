# THE SYRIATEL CUSTOMER CHURN ANALYSIS
## Overview of the project
Customer churn is a critical issue for businesses, especially in highly competitive industries like telecommunications. Churn occurs when customers stop using a company’s services and switch to competitors. Retaining customers is often more cost-effective than acquiring new ones, which makes churn prediction an important task.

In response to this challenge, telecom companies are exploring churn prediction mechanisms, which will proactively address customer concerns, improve service delivery, and implement targeted retention strategies.  This project aims to develop a predictive model that will identify customers at risk of churning, helping SyriaTel Telecommunications company minimize churn and reduce losses.
## Business and Data Understanding 
Business problem

Syriatel is struggling with rising customer churn, which threatens its revenue and market position. The company lacks a clear way to know customers that are likely to leave. Without timely insights, Syriatel risks losing more customers to competitors and incurring higher costs to replace them. Developing a predictive solution is critical to identify at risk customers early and take proactive steps to retain them.

Data Understanding

This project utilizes a publicly available [Syriatel Telecom Customer Churn Dataset](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset) from Kaggle containing information about Syriatel's customers. The dataset contains 3333 customer records and 21 features therefore, it is ideal for this task because it contains a demographic, account, and service usage information that is typical in the telecom industry and highly relevant for predicting churn.
The key features for the dataset were:

1. `State` and `Area code`: It contains the different states and area codes of the customers subscribed to SyriaTel company, both who are churning and who are not churning from the company.

2. `International plans` and `Voice Mail Plans`: It gives directive as to which customers are subscribed to either an International plan, voice mail plan, both, or even none.

3. `Call rates`: It gives information on the different call rates for the customers for day, night, evening, and international calls

4. `Customer Service calls`: It provides information on the number of customer service calls the customers are receiving from support staff in the company.
## Data Preparation
Data Cleaning

Data cleaning is a crucial step to prepare the dataset for reliable analysis and modeling. The goal is to remove inconsistencies, handle missing, handle duplicate values, and standardize formats so that the data is accurate, consistent and ready for further exploration. It involved:

1. Handling Missing Values: The dataset was verified to have no missing values. 

2. Handling Duplicates: No duplicates were found, ensuring each row represented a unique customer.

3. Encoding : Categorical string values were standardized into a numerical format to make them suitable for machine learning algorithms.

4. Feature Engineering & Selection: The phone number was dropped as it was  a unique identifier and not a predictive feature.

5. To avoid multicollinearity, the highly correlated minutes columns e.g., total day minutes were removed. 

6. Outlier Treatment: We retained outliers as they reflected customer behaviours.

7. Feature Scaling: Finally, all numerical features were standardized using StandardScaler. 

Exploratory Data Analysis

We generating visualizations and plots to examine relationships between the features.

Key Visualizations
## Modeling approach
The problem is framed as a supervised binary classification task. Our steps were:

1. Implementing the train_test_split method to split the dataset into training and testing sets (80/20 split)

2. Resampling the data to handle class imbalance in the target variable.

3. Training four different types of models to measure performance using recall and ROC-AUC metrics, with Logistic Regression as the baseline model.

4. Predictions were evaluated against the true test values (y_test) using a comprehensive set of metrics.

5. Plotting the confusion matrix to evaluate the rate of true and false positives and negatives for each model.

## Evaluation
We assessed how well our models performed and determined whether they met the business objectives of predicting and reducing customer churn.

Model Comparison – Key Metrics

We compared Logistic Regression, Random Forest, Decision Tree, and K-Nearest Neighbors (KNN) using accuracy, precision, recall, and F1-score.

 - Random Forest achieved the highest performance with 91.9% accuracy and an F1-score of 70%, making it the most reliable model for churn prediction.

 - Decision Tree performed moderately with balanced recall and accuracy.

 - Logistic Regression and KNN achieved similar results but struggled with precision, making them less suitable for deployment.

Model Comparison – ROC Curves

The ROC curves and AUC scores further highlighted Random Forest as the best-performing model. Its curve stayed closest to the top-left corner of the plot, indicating strong discriminatory power between churners and non-churners.

Hyperparameter Tuning – Random Forest

We optimized the Random Forest model using GridSearchCV with cross-validation. After tuning the model achieved 91% accuracy on the test set.It correctly identified 71% of churners (recall) while maintaining 69% precision, balancing the trade-off between false positives and false negatives.

Feature importance analysis revealed that customer service calls, international plans, and day-time charges were the strongest predictors of churn.

Key Visualizations
1. The model metrics comparison
![Metrics](Images/metrics.png)

2. The ROC curve
![ROC Comparisons](Images/roc.png)

3. Feature Importances
![Features Importance](Images/features.png)
## Conclusion
The analysis of the Syriatel customer churn dataset provided valuable insights into the factors that drive customer churn. Through exploratory analysis and predictive modeling, we identified customer service calls, international plans, and day time charges as the strongest indicators of churn.

Among the models tested, the Random Forest Classifier consistently outperformed others, achieving 91% accuracy and correctly identifying 71% of churners after hyperparameter tuning. This balance between accuracy and recall makes it a practical tool for churn prediction in real-world scenarios.

Recommendations:

Enhance customer support services to reduce complaints and frustrations.

Reassess pricing strategies for international and high-usage plans.

Implement targeted retention campaigns focused on customers flagged as high risk by the churn model.

Continuously monitor churn metrics and update the predictive model with new customer data to maintain accuracy.