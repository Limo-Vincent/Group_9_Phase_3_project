# Group_9_Phase_3_project
## SyriaTel Customer Churn Prediction

## Introduction  
Customer churn is a critical issue for businesses, especially in highly competitive industries like telecommunications. Churn occurs when customers stop using a company’s services and switch to competitors. Retaining customers is often more cost-effective than acquiring new ones, which makes churn prediction an important task.  

In this project, we developed and compared different machine learning models to predict whether a customer is likely to churn. The goal was not only to build accurate models but also to gain insights into the factors that influence customer churn. This knowledge can help businesses take proactive measures to retain their customers.  

## Objectives  
The overall objective of this project was to build a machine learning model that accurately predicts customer churn.  

Specific objectives were:  
1. To determine how customer demographics influence churn.  
2. To examine the effect of service usage on churn.  
3. To analyze the impact of financial factors on churn.  
4. To identify behavioral indicators that signal potential churn.  
5. To segment customers and assess which groups are most at risk of churning.  

## Data Preparation and Exploration  
The dataset contained customer demographic details, service usage statistics, billing information, and churn labels. Before modeling, we conducted data cleaning and preprocessing, which included:  
- Handling categorical features such as state and service plans.  
- Encoding categorical variables into numeric form.  
- Checking for missing values and ensuring data consistency.  
- Splitting the dataset into training and test sets for model evaluation.  

We also carried out Exploratory Data Analysis (EDA):  
- **Univariate analysis** was used to study individual features (for example, distributions of call charges and service plans).  
- **Bivariate analysis** helped us compare features against churn, highlighting relationships such as higher churn among customers with international plans.  
- **Correlation analysis** was performed on numerical features to identify patterns and dependencies between variables.  

Because the dataset had an imbalance (fewer churn cases than non-churn cases), we applied **SMOTE (Synthetic Minority Oversampling Technique)**. This technique generates synthetic samples for the minority class, helping the models learn patterns of churn more effectively.  

## Models Used  
We trained and compared several machine learning models:  
- **Logistic Regression** – A baseline linear model useful for interpretability.  
- **Decision Tree** – Captures non-linear relationships and provides easy-to-understand rules.  
- **Random Forest** – An ensemble method that combines multiple decision trees, offering robustness and higher accuracy.  
- **K-Nearest Neighbors (KNN)** – Included to test a neighbor-based approach, which classifies customers based on similarity to others. KNN is particularly useful in detecting churn patterns among customers with similar service usage or demographic profiles.  

## Findings  
From the analysis and modeling, the following key findings were observed:  
1. **Demographic factors** such as having an international plan were strongly linked to higher churn rates.  
2. **Service usage** played a key role: customers who made more frequent customer service calls had a higher probability of churning.  
3. **Financial variables** like total day charges and international charges were significant predictors of churn, showing that billing patterns influenced customer decisions.  
4. **Behavioral patterns** such as repeated service issues emerged as strong churn signals.  
5. The **ensemble models (Random Forest)** and **instance-based models (KNN)** performed better than the simpler Logistic Regression model, highlighting the need for more complex methods to capture the underlying patterns in the data.  

## Recommendations  
Based on the findings, the following recommendations were made:  
1. Customers with international plans or higher billing charges should be given special attention, possibly through tailored retention offers.  
2. Frequent callers to customer service should be flagged early for retention efforts, as this behavior strongly correlates with dissatisfaction.  
3. Financial monitoring of charges should be prioritized, and pricing strategies should be reviewed to prevent customer dissatisfaction.  
4. Behavioral data such as repeat complaints should be integrated into monitoring systems to identify customers at risk of leaving.  
5. Random Forest and KNN models are suitable candidates for deployment, as they capture complex and similarity-based churn patterns better than linear models.  

## Conclusion  
This project demonstrated the effectiveness of machine learning models in predicting customer churn. The analysis showed that churn is influenced by a combination of demographic, service, financial, and behavioral factors. By using advanced models, businesses can better identify at-risk customers and design targeted retention strategies.  

Future work could involve testing additional ensemble methods like Gradient Boosting or XGBoost, incorporating more real-time behavioral data, and deploying the model in a live environment where it can provide actionable churn alerts to the business team.  

Author
Marion Mengich
