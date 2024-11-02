# [Predicting Credit Card Approvals (Classification - Supervised Learning)](https://www.datacamp.com/datalab/w/4cdd5468-4215-407f-b0de-c5b6dbab903e/edit) 👈
### Overview
![Approval_image](https://github.com/user-attachments/assets/ea03e11e-cd58-4582-9168-5a2ef257189f)

Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this workbook, you will build an automatic credit card approval predictor using machine learning techniques, just like real banks do.</p>
In this project, I explore a machine learning classification problem focused on predicting credit card approval status based on applicant information. The dataset includes 690 samples with features such as demographic information, financial status, and employment history.</p>

### Dataset Description

| Old Column Name | Column Name    | Description                     | Data Type |
|-----------------|----------------|---------------------------------|-----------|
| 0               | gender         | Gender of the applicant         | Object    |
| 1               | age            | Age of the applicant            | Object    |
| 2               | debt           | Total debt                      | Float64   |
| 3               | married        | Marital status                  | Object    |
| 4               | bankCustomer   | Bank customer status            | Object    |
| 5               | educationLevel | Education level                 | Object    |
| 6               | ethnicity      | Ethnicity                       | Object    |
| 7               | yearsEmployed  | Years of employment             | Float64   |
| 8               | priorDefault   | History of previous defaults    | Object    |
| 9               | employed       | Employment status               | Object    |
| 10              | creditScore    | Credit score                    | Int64     |
| 11              | citizen        | Citizenship status              | Object    |
| 12              | income         | Annual income                   | Int64     |
| 13              | approvalStatus | Approval status (target)        | Object    |
Dataset from the UCI Machine Learning Repository.

### Project Objectives
The main goal of this project is to build a machine learning model that can accurately predict the approvalStatus of credit card applications based on the provided features. I aimed to:

1. Perform Exploratory Data Analysis (EDA) to understand the numerical and categorical features of the dataset.
2. Preprocess the data by handling missing values and encoding categorical variables.
3. Apply feature scaling to enhance model performance.
4. Test multiple classification algorithms to identify the best-performing model.
5. Evaluate and optimize model accuracy with metrics such as precision, recall, and F1-score.

### Methods
#### 1. Exploratory Data Analysis (EDA)
##### 1.1 Data Cleaning
- First, we reviewed the dataset to identify missing values and found that it did not contain null values but had entries marked with a '?' symbol, likely indicating missing information. For simplicity, we removed any records with these placeholders and changed the data type of the 'age' column from Object to Float64.
```python
cc_apps.drop(cc_apps[cc_apps.isin(["?"]).any(axis=1)].index, inplace=True)
cc_apps["age"] = cc_apps["age"].astype("float")
cc_apps.info()
```
- The dataset columns were renamed for clarity (e.g., "age," "debt," "education level") to facilitate interpretation and analysis.
![image](https://github.com/user-attachments/assets/1139f692-3c29-4b6a-b2ec-1f70a04d5a69)

##### 1.2 Feature Analysis
- The data in numerical (e.g. debt, years employed). Using describe() functions, we gained a high-level summary of the data distribution and feature types.
```python
numerical_columns.describe()
```
![image](https://github.com/user-attachments/assets/c8cab214-b4a2-4d92-b679-5060c3b4d2fb)
I found that some numeric features have a high standard deviation; therefore, for the performance of the model, we need to standardize them.

##### 1.3 Visualization
- We used histograms to visualize the distribution of numerical features, which indicated skewness in age, debt, income, years employed, and credit score.
```python
numerical_columns.hist(figsize=(12,12),bins=20)
```
![image](https://github.com/user-attachments/assets/e79324ae-85f2-4833-b446-6803bcf6279c)
![image](https://github.com/user-attachments/assets/a547e165-5f4f-4bfd-adb8-3ac755306900)
![image](https://github.com/user-attachments/assets/d4b341bc-30ef-46d7-acd0-bb4002a77df5)

- Box plots were employed to examine the relationship between approval status and each numerical variable, highlighting outliers and trends in features like debt and years employed.
```python
for column in numerical_columns.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=cc_apps["approvalStatus"], y=cc_apps[column], showfliers=False, palette=['green', 'red'])
    plt.title(f'Boxplot of {column} by Approval Status')
    plt.xlabel("Approval Status")
    plt.ylabel(column)
    plt.show()
```
![image](https://github.com/user-attachments/assets/25e716c4-0d90-44b8-8c06-7b0138701a3b)
![image](https://github.com/user-attachments/assets/45f1bc42-f5f7-41b0-a899-c23d058cc02d)
![image](https://github.com/user-attachments/assets/bdb72f9a-2426-436d-ab38-b15e38a8d009)
![image](https://github.com/user-attachments/assets/77113485-40c4-439b-86b8-d27b7343901a)
![image](https://github.com/user-attachments/assets/0a9f179f-ffff-487f-91a6-47211b2fd754)
<p>I found that years of employment, credit score, and income impact the changes in credit card approval, as shown in the boxplot above.</p>

- Count plots were created for categorical variables to observe the class distribution across approval statuses, providing insight into the proportion of approved and declined applications for each category (e.g., gender, education level).
```python
for col in categorical_columns.columns:
    print(f"\033[1m{col}\033[0m\n")
    plt.figure(figsize=(8, 6))
    sns.countplot(data=cc_apps, x=col, hue="approvalStatus", palette=['green', 'red'])
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f'Count of {col} by Approval Status')
    plt.legend(title="Approval Status")
    plt.show()
```
![image](https://github.com/user-attachments/assets/0164cda9-8687-4820-8003-be92b8c3e3e5)

In the categorical features, I gained insights into how they affect approval or decline rates per category. However, I would like to highlight the outstanding feature that clearly differentiates between categories regarding approval or decline: the history of previous defaults.

- Lastly, we generated a correlation heatmap to understand the relationships among numerical features, which revealed mild correlations between features like age and years employed.
```python
import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cc_apps.corr(), ax=ax, annot=True, cmap="crest")
plt.show()
```
![image](https://github.com/user-attachments/assets/4bfba638-6c87-4baa-8949-bbfdaac4545b)

#### 2. Modeling
##### 2.1 Data Preprocessing
- Encoding techniques were used to convert categorical features to numerical representations, necessary for model training.
```python
cc_apps_encoded = pd.get_dummies(cc_apps, drop_first=True)
cc_apps_encoded.head()
```
![image](https://github.com/user-attachments/assets/a2bfd332-eb45-4809-94e9-8ea2430fd3dc)

- Define 𝑋 as the features and 𝑦 as the target variable that I want to predict. I will split the data into training and testing sets to reduce overfitting in the model.
```python
X = cc_apps_encoded.drop("approvalStatus", axis=1)
y = cc_apps_encoded["approvalStatus"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
```
- Prepare standardization methods to apply to the numerical features in order to maintain a consistent scale and achieve optimal performance. Additionally, I will compile a list of classification models to use for model selection.
```python
scaler = StandardScaler()
minmaxscaler = MinMaxScaler()
models = {"Logistics Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree": DecisionTreeClassifier()}
result_scaler = []
result_minmaxscaler = []
```
##### 2.2 Model Selection
- We experimented with multiple classification algorithms—Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree Classifier—applying each to the different scalers.
```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for model in models.values():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    result_scaler.append(cv_results)

plt.boxplot(result_scaler, labels=models.keys())
plt.show()
```
![image](https://github.com/user-attachments/assets/1aa56b5d-f516-46ed-9a51-10794eadcf04)
```python
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    test_score = model.score(X_test_scaled, y_test)
    print("{} Test Set Accuracy: {}".format(name, test_score))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(conf_matrix)
```
![image](https://github.com/user-attachments/assets/0b986e5f-9f7d-4aea-9afb-d4126353036d)
```python
X_train_min_max_scaled = minmaxscaler.fit_transform(X_train)
X_test_min_max_scaled = minmaxscaler.transform(X_test)

for model in models.values():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_min_max_scaled, y_train, cv=kf)
    result_minmaxscaler.append(cv_results)

plt.boxplot(result_minmaxscaler, labels=models.keys())
plt.show()
```
![image](https://github.com/user-attachments/assets/368675b7-9c17-4cee-9ca9-02b1b52eecc3)
```python
for name, model in models.items():
    model.fit(X_train_min_max_scaled, y_train)
    y_pred = model.predict(X_test_min_max_scaled)
    test_score = model.score(X_test_min_max_scaled, y_test)
    print("{} Test Set Accuracy: {}".format(name, test_score))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(conf_matrix)
```
![image](https://github.com/user-attachments/assets/38c1112c-f4dc-4c80-813b-de4d09864444)

From the data above, the best scaler among two is StandardScaler and the most effective classification algorithm among the three is Logistic Regression. Next, we will use both StandardScaler and Logistic Regression in a pipeline to perform hyperparameter tuning to achieve optimal performance and identify the best parameters.

- Hyperparameter tuning was performed using GridSearchCV to optimize model parameters.
```python
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


param_grid = {
    'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'logisticregression__tol': [0.01, 0.001, 0.0001],
    'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__solver': ['liblinear', 'saga'], 
    'logisticregression__max_iter': [100, 200, 300, 400, 500]           
}

pipe = make_pipeline(StandardScaler(), LogisticRegression())
grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
```
![image](https://github.com/user-attachments/assets/e1ea48de-2a2e-4aa6-bc72-2fe1ba2dce81)

##### 2.3 Evaluation
- Models were evaluated based on accuracy, precision, recall, and F1 score, with cross-validation to ensure robust results across different subsets of data.
```python
print("Best cross-validated score: {:.2f}".format(grid_search.best_score_))
```
![image](https://github.com/user-attachments/assets/3b932802-fb42-4f0e-8e43-a7f06fe49fc2)
- Confusion matrices were used to visualize each model’s classification accuracy for approved vs. declined applications.
```python
y_pred = grid_search.predict(X_test)
best_score = accuracy_score(y_test, y_pred)
print('Test set accuracy: {:.2f}'.format(best_score))
print(grid_search.score(X_test, y_test))
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
class_report = classification_report(y_test, y_pred)
print('Classification Report')
print(class_report)
```
![image](https://github.com/user-attachments/assets/9ef0ea5b-c8e4-41e5-9219-515ddfe373b7)

Best accuracy for test set is 0.89

### Results
- The Logistic Regression performed best, achieving an accuracy score of **_89%_**, followed by KNN and Decision Tree.

| Metric       | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|--------------|---------|---------|-----------|--------------|
| Precision    | 0.98    | 0.81    | 0.90      | 0.90         |
| Recall       | 0.81    | 0.98    | 0.89      | 0.89         |
| F1-Score     | 0.89    | 0.89    | 0.89      | 0.89         |
| Support      | 72      | 60      | 132       | 132          |
| Accuracy     | 0.89    |         |           |              |

- Key features contributing to credit approval prediction included years of employment, credit score, and income, suggesting these factors play a significant role in approval likelihood.

### Conclusion
This project successfully automated the credit card approval process using machine learning, providing insights into critical features affecting approvals.

### Next Steps
- **_Explore Additional Algorithms:_** Investigate and implement alternative machine learning algorithms to improve model accuracy and performance, such as ensemble methods or deep learning approaches.
- **_Analyze Categorical Features:_** Conduct a deeper study of categorical features to evaluate their correlation with the target variable.
- **_Deployment Considerations:_** Plan for the deployment of the model in a real-world environment, considering aspects such as scalability, monitoring, and maintenance.
