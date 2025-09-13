# Group_9_Phase_3_project
## SyriaTel Customer Churn Prediction

This project aims to develop a prediction model for SyriaTel Telecommunication company, that will be able to predict customer churn from the company.

## 💼 Business Understanding

### Introduction

The telecommunications industry has become very competitive over the years, with customer retention emerging as a critical challenge. One of the major issues facing telecom providers is customer churn - a scenario where users discontinue their service, either due to dissatisfaction from the provider, or due to the availability of better alternatives. High churn rates can significantly impact a company's overall revenue, and scaling potential.

In response to this challenge, telecom companies are exploring churn prediction mechanisms, which will proactively address customer concerns, improve service delivery, and implement targeted retention strategies. In light of this, this project aims to develop a predictive model that will identify customers at risk of churning, helping SyriaTel Telecommunications company minimize churn, and enhance long-term profitability Enable data-driven decision-making in customer retention strategies.

### Business Problem

SyriaTel, a leading telecom provider, is experiencing a significant loss of customers who are choosing to leave its services for other competitors. To address this challenge, the company seeks to build a robust predictive model capable of identifying customers who are at risk of churning. By leveraging on data-driven insights and predictive modeling, SyriaTel aims to understand the key drivers of customer attrition, determing methods of improving long-term retention of customers, and enhance long-term customer loyalty and profitability.

## 📊 Dataset
This analysis and modeling utilizes the Telecom Churn Dataset from Kaggle. The dataset includes essential customer-churn attributes such as:

State and Area code: It contains the different states and area codes of the customers subscribed to SyriaTel company, both who are churning and who are not churning from the company.

International and Voice Mail Plans: It gives directive as to which customers are subscribed to either an International plan, voice mail plan, both, or even none.

Call rates: It gives information on the different call rates for the customers for day, night, evening, and international calls

Customer Service calls: It provides information on the number of customer service calls the customers are receiving from support staff in the company.

## 🔍 Methodology
To ensure an effective data analysis and predictive modeling strategy, the notebook will follow a structured approach:
### 1️⃣ Data Exploration
Loading of the dataset to analyze columns in the dataset, along with data types and number of records.

Checking for missing values, duplicates or any inconsistencies in the data.

Computing the descriptive statistics of the dataset to get an idea of the key statistical attributes.

### 2️⃣ Data Manipulation
Standardizing the column names to have the same name formatting.

Converting the Area_Code feature to an object.

Dropping the Phone_Number feature from the dataset as it was not a major requirement in my analysis and modeling.

### 3️⃣ Exploratory Data Analysis
Generating visualizations and plots to examine relationships between the features (Univariate & Bivariate Analysis).

Investigating the relationship between the features and the target (Churn) through visualizations such as distribution plots, correlation heatmaps, box plots and bar plots.

Handling outliers in the dataset.

Dropping columns that have high multicollinearity

### 4️⃣ Data Preprocessing
Encoding the features in the dataset for ease of implementation in the modeling stage through techniques such as Label Encoding and One-Hot Encoding.

Scaling the numerical features to a range of (0, 1) using MinMaxScaler

### 5️⃣ Predictive Modeling
Resampling the data to handle class imbalance in the target variable.

Implementing the train_test_split method to split the dataset into training and testing sets (80/20 split)

Training six different types of models to measure performance using recall and ROC-AUC metrics, with Logistic Regression as the baseline model.

Plotting the confusion matrix to evaluate the rate of true and false positives and negatives for each model.

### 6️⃣ Model Evaluation
Plotting ROC curves and computing the AUC scores of all the six models, and doing a comparison of the curves and the AUC scores.

Computing the recall score of all the six models and comparing the scores of each model.

Determining the important features for the best performing model.

### 7️⃣ Conclusions and Business Recommendations
Drawing conclusions from the analysis and modeling process.

Provision of data-driven insights and recommendations, based on the conducted analysis.

## key Visualization
1. ROC Curve 
![ROC](Images/roc.png)

## 💼 Business Recommendations
1.Enhance Customer Service Efficiency: A high amount of customer service interactions with customers is seen to increase churn. Investing in comprehensive training sessions for support stuff, and implmenting better issue/conflict resolution frameworks can significantly boost customer satisfation, and in turn minimize the rate of customer churn.

2.Review and Optimize Call Rate Plans: A large proportion of churners are linked to high day, evening, night, and international call charges. Reassessing the current pricing structure and introducing more competitive or bundled rate plans will make services more attractive and cost-effective, reducing churn caused by price dissatisfaction

3.Targeted Incentives for High-Churn Area Codes: Customers in area codes 415 and 510 show a higher likelihood of churn. Introducing specialized incentives such as discounts, loyalty rewards, or exclusive promotions for these regions can help retain at-risk customers and strengthen long-term relationships.

4.State-Specific Retention Strategies: States such as Texas, New Jersey, Maryland, Florida (Miami), and New York exhibit above-average churn rates. Developing localized retention strategies, including personalized engagement, region-specific promotions, and enhanced customer support, will help strengthen loyalty in these high-risk markets.

## 📶 Next Steps
Model Improvement
Hyperparameter Optimization: Apply Grid Search (or Randomized Search) to fine-tune hyperparameters for better performance and generalization.

Handling Class Imbalance: Explore resampling techniques such as class weights, ADASYN, or ensemble methods to address imbalance and improve churn prediction.

Feature Engineering: Derive new variables (e.g., customer lifetime value, call pattern ratios) to increase model robustness and predictive power.

Preprocessing Pipelines: Implement automated preprocessing pipelines (scaling, encoding, feature selection) to reduce data leakage and ensure consistent transformations across training and production.

## 📝 Conclusion
From our prediction modeling analysis, The K nearest model had a recall score of 0.40, while the Gradient Boosting model achieved a recall score of 0.82. However, the Gradient Boosting model had a higher AUC score of 0.921, while the Random forest model had an AUC score of 0.911. We were able to meet all our set objectives, which were to build a customer churn prediction model with a recall score of 0.8 and above, and to identify the key features that contribute significantly to customer churn, which include Customer Service Calls, Total Day Charge, and International Plan. Due to the nature of the project and the prediction problem, I would recommend the XGBoost classifier model with a higher recall for predicting customer churn rates at SyriaTel Telecommunication company.

Author: Dennis Chesire
Email:denniskipropchesire@gmail.com
LinkedIn:www.linkedin.com/in/dennis-chesire-780003294
