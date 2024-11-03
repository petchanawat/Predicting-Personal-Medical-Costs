# [Predicting Personal Medical Costs (Regression - Supervised Learning)](https://colab.research.google.com/drive/17gY7U5w77YlPW_L5fMpOwZLCVx7KEH_l?usp=sharing) 👈
### Overview
![image](https://github.com/user-attachments/assets/9348755b-2f3a-4442-b21a-1248a06a1c59)

This project focuses on predicting an individual's medical expenses using machine learning. Medical costs can be influenced by various factors, including age, sex, BMI, number of children, smoking status, and region. An accurate predictive model can help in budgeting and planning for future medical expenses. 
<p>This project uses regression techniques to predict medical charges based on personal and demographic information. The dataset includes 1338 samples with features such as age, health information.</p>

### Dataset Description

| Column Name | Description | Data Type |
|-------------|-------------|-----------|
| age         | Age of the primary beneficiary | int64 |
| sex         | Gender of the insurance contractor: female or male | object |
| bmi         | Body mass index (BMI), indicating body weight relative to height (kg/m²); ideal range: 18.5 to 24.9 | float64 |
| children    | Number of children or dependents covered by health insurance | int64 |
| smoker      | Smoking status of the individual (yes/no) | object |
| region      | Residential area of the beneficiary in the US: northeast, southeast, southwest, or northwest | object |
| charges     | Medical costs billed to the individual by health insurance | float64 |
Dataset source: [Kaggle Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance/data).

### Project Objectives
The primary goal of this project is to build a regression model that can accurately predict an individual’s medical charges based on their characteristics. The steps include:

1. Performing Exploratory Data Analysis (EDA) to examine the dataset.
2. Preprocessing the data by encoding categorical variables.
3. Applying feature scaling to improve model performance.
4. Testing multiple regression algorithms to identify the best-performing model.
5. Evaluating and optimizing model accuracy with metrics such as R-squared, Mean Squared Error, Mean Absolute Error, and Root Mean Squared Error.

### Methods
#### 1. Exploratory Data Analysis (EDA)
##### 1.1 Data Cleaning
- No missing values were found, so the dataset did not require any imputation.
```python
df.info()
```
![image](https://github.com/user-attachments/assets/3a2d9856-023f-4b6d-b59c-e42182c117cb)

- The dataset columns have the correct data types, and the categorical columns do not contain any unusual categories.
![image](https://github.com/user-attachments/assets/d8c6fc80-ac32-4a67-b15f-4d78a3c75084)

##### 1.2 Feature Analysis
- Descriptive statistics of numerical columns (age, bmi, children, charges) were computed to understand distributions.
```python
numerical_columns.describe()
```
![image](https://github.com/user-attachments/assets/09266ccb-99b5-46d9-a6a5-d196afcaa3bd)

I found that targets have a high standard deviation; therefore, for the performance of the model, we need to transform it.

##### 1.3 Visualization
- Histograms illustrated the distributions of numerical features such as bmi and charges.
```python
numerical_columns.hist(figsize=(12, 12))
```
![image](https://github.com/user-attachments/assets/dcf30514-a212-487f-9b35-790e14aa028c)

- Box plots highlighted outliers of numerical features.
```python
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(df[column])
    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)
    plt.show()
```
![image](https://github.com/user-attachments/assets/fa94291e-d01f-4c3c-867a-933c480ae90a)
![image](https://github.com/user-attachments/assets/4ff128c1-5b89-4446-a31c-f632510b1700)
![image](https://github.com/user-attachments/assets/7d43fdcd-feaa-400c-bb19-83d3d135a9fa)
![image](https://github.com/user-attachments/assets/53980e44-5c8f-4490-b580-0dd40e64b759)

I found outliers in the bmi and charges columns, particularly in charges.

- Calculate the interquartile range (IQR) to handle outliers, ensuring they can be removed if they do not exceed 5% of the data.
```python
bmi_25, bmi_75 = np.percentile(df["bmi"], [25, 75])
charges_25, charges_75 = np.percentile(df["charges"], [25, 75])
bmi_iqr = bmi_75 - bmi_25
charges_iqr = charges_75 - charges_25

outlier_ratio_bmi = len(df[(df["bmi"] > (np.percentile(df["bmi"], 75) + 1.5 * bmi_iqr)) | (df["bmi"] < (np.percentile(df["bmi"], 25) - 1.5 * bmi_iqr))])/len(df)
outlier_ratio_charges = len(df[(df["charges"] > (np.percentile(df["charges"], 75) + 1.5 * charges_iqr)) | (df["charges"] < (np.percentile(df["charges"], 25) - 1.5 * charges_iqr))])/len(df)
print("%Outlier ratio of BMI per all data: {:.3f}%".format(outlier_ratio_bmi*100))
print("%Outlier ratio of Charges per all data: {:.3f}%".format(outlier_ratio_charges*100))
```
![image](https://github.com/user-attachments/assets/bc677972-ec9b-4b14-84c2-40a3455952d0)
```python
df.drop(df[(df["bmi"] > (np.percentile(df["bmi"], 75) + 1.5 * bmi_iqr)) | (df["bmi"] < (np.percentile(df["bmi"], 25) - 1.5 * bmi_iqr))].index, axis=0, inplace=True)
```
Dropped outliers in the bmi data, as they account for only 0.67% of all rows. However, retained outliers in charges since they represent about 10% of the data, and removing them could significantly impact the training set.
- Examine the linear correlation between numerical features and the target variable using scatter plots to visualize trends.
```python
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    plt.plot(df[column], df["charges"], marker=".", linestyle='none')
    plt.title(f"Line plot of charges by {column}")
    plt.xlabel(column)
    plt.ylabel("Charges")
    plt.show()
```
![image](https://github.com/user-attachments/assets/98f7305f-4ca8-427f-be93-619431a4b35e)
![image](https://github.com/user-attachments/assets/7fb571ad-6b98-4c2e-96e0-b68ea39eb6eb)
![image](https://github.com/user-attachments/assets/808cf730-c575-4a2d-835d-e0ac7727b715)

I observed that the 'children' column contains discrete values and should be classified as an object type for encoding during preprocessing.
```python
df["children"] = df["children"].astype("object")
df.info()
```

- Create a boxplot to visualize charges for each categorical features, such as charges by male and female sex, and a countplot to display the number of observations for each category.
```python
categorical_columns = df.select_dtypes(include='object')
for column in categorical_columns.columns:
  plt.figure(figsize=(8, 6))
  sns.boxplot(x=df[column], y=df["charges"])
  plt.title(f'Box plot of charges by {column}')
  plt.xlabel(column)
  plt.ylabel("Charges")
  plt.show()
```
![image](https://github.com/user-attachments/assets/4ddd2cab-664f-4d43-8574-ae7d3783d2f1)
![image](https://github.com/user-attachments/assets/0c063fa6-2ad2-4e20-9187-716c9ee18cfa)
![image](https://github.com/user-attachments/assets/cdcdaa70-d1d0-4798-b0e7-1f837641f3f3)
![image](https://github.com/user-attachments/assets/29286746-a3cc-4eab-b0c5-821880a77a0f)

It was found that smoking status significantly impacts medical expenses.

```python
for col in categorical_columns.columns:
  print(f"\033[1m{col}\033[0m\n")
  plt.figure(figsize=(8, 6))
  sns.countplot(data=df, x=col)
  plt.xlabel(col)
  plt.ylabel("Count")
  plt.title(f"Count of {col}")
  plt.show()
```
![image](https://github.com/user-attachments/assets/efc5a642-5d8e-458e-a9c4-8b69e63a47b1)
![image](https://github.com/user-attachments/assets/a337ae9b-83be-4284-baf9-7b92cb32ea2d)
![image](https://github.com/user-attachments/assets/c8c5ff63-8712-44d7-9099-328658dcb987)
![image](https://github.com/user-attachments/assets/9830f514-47ab-4665-9475-3827bbeae9ff)

- Lastly, we generated a correlation heatmap to understand the relationships among numerical features, which revealed mild correlations between features like age and bmi.
```python
numerical_columns = df.select_dtypes(include='number')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numerical_columns.corr(), ax=ax, annot=True, cmap="crest")
plt.show()
```
![image](https://github.com/user-attachments/assets/ddc173fa-b2b5-48fb-a209-51279a414cce)

#### 2. Modeling
##### 2.1 Data Preprocessing
- Encoding techniques were used to convert categorical features to numerical representations, necessary for machine learning.
```python
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded.info()
```
![image](https://github.com/user-attachments/assets/61be8377-3435-4995-ac98-7bedc1d6e7c1)

- Various functions were defined to transform the target variable due to its high standard deviation, including square root and logarithmic transformations.
```python
def no_transform(y):
    return y

def sqrt_transform(y):
    return np.sqrt(y)

def log_transform(y):
    return np.log1p(y) 

def some_sqrt_transform(y):
    return np.sqrt(y + 1) 

def some_log_transform(y):
    return np.log1p(y + 1) 

# List of transformation functions
transformations = {
    'No Transformation': no_transform,
    'Square Root': sqrt_transform,
    'Log Transformation': log_transform,
    'Some Square Root': some_sqrt_transform,
    'Some Log': some_log_transform
}
```

- Define 𝑋 as the features and 𝑦 as the target variable that I want to predict.
```python
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]
```
- Standardization methods, including Robust Scaler and Standard Scaler, are prepared to apply to numerical features, ensuring a consistent scale and enhancing model performance. Additionally, a set of regression models is compiled for model selection.
```python
robust_scaler = RobustScaler()
scaler = StandardScaler()

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "Support Vector Regressor": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "MLP Regressor": MLPRegressor(),
    "XGBoost": XGBRegressor(),
    "LightGBM": LGBMRegressor()
}
```

##### 2.2 Model Selection
- The code applies both RobustScaler and StandardScaler to numerical features, splits the data, and evaluates a set of regression models, identifying the best model based on R² for each scaling method. The top model and corresponding metrics for each scaler are displayed, with a focus on "No Transformation" applied to the target variable.
```python
results_robust = {}
results_standard = {}

y_no_transform = no_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_no_transform, test_size=0.2, random_state=42)

robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)

for name, model in models.items():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)

    mse_cv_results = cross_val_score(model, X_train_robust, y_train, cv=kf, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_cv_results)  

    r2_cv_results = cross_val_score(model, X_train_robust, y_train, cv=kf, scoring='r2')
    r2 = np.mean(r2_cv_results)

    results_robust[name] = {
        "MSE": mse,
        "Std dev": np.std(mse_cv_results),
        "R²": r2
    }

standard_scaler = StandardScaler()
X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)

for name, model in models.items():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)

    mse_cv_results = cross_val_score(model, X_train_standard, y_train, cv=kf, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_cv_results)  

    r2_cv_results = cross_val_score(model, X_train_standard, y_train, cv=kf, scoring='r2')
    r2 = np.mean(r2_cv_results)

    results_standard[name] = {
        "MSE": mse,
        "Std dev": np.std(mse_cv_results),
        "R²": r2
    }

def get_best_model(results):
    best_model_name = max(results, key=lambda k: results[k]['R²'])
    return best_model_name, results[best_model_name]

best_robust_model_name, best_robust_metrics = get_best_model(results_robust)

best_standard_model_name, best_standard_metrics = get_best_model(results_standard)

print("Best model using RobustScaler:")
print(f"{best_robust_model_name}: MSE = {best_robust_metrics['MSE']}, Std dev = {best_robust_metrics['Std dev']}, R² = {best_robust_metrics['R²']}")

print("\nBest model using StandardScaler:")
print(f"{best_standard_model_name}: MSE = {best_standard_metrics['MSE']}, Std dev = {best_standard_metrics['Std dev']}, R² = {best_standard_metrics['R²']}")
```
![image](https://github.com/user-attachments/assets/b1b51a09-bb1d-4c13-bdbf-feb16c499604)

- For each target transformation—no transformation, square root, and logarithm—the code applies both RobustScaler and StandardScaler to the numerical features. The model and scaler combination with the highest R² and lowest MSE is selected for optimal performance. The best result is achieved with Gradient Boosting, yielding an MSE of 0.1389, a standard deviation of 0.0392, and an R² of 0.8375.
![image](https://github.com/user-attachments/assets/5465dee1-b3b3-457e-96b9-c7039bba95fd)

Use this scaler and model to find best parameters and scores.

##### 2.3 Model Optimization
- Determine the importance of each feature influencing the target variable.
```python
y_log_transform = log_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_log_transform, test_size=0.2, random_state=42)
standard_scaler = StandardScaler()
X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train_standard, y_train)

feature_importances = gb_model.feature_importances_
features = X_train.columns 

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Gradient Boosting Regressor')
plt.gca().invert_yaxis()
plt.show()

print(importance_df)

```
![image](https://github.com/user-attachments/assets/83561521-71af-4654-9b83-83b1e2eb939a)

The top three most important features are, in order: smoker, age, and BMI.

- Hyperparameter tuning was performed using GridSearchCV to improve the Gradient Boosting model, as it showed promising initial results. The model was further refined by adjusting the learning rate, number of estimators, and depth.
```python
from sklearn.pipeline import make_pipeline
y_log_transform = log_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_log_transform, test_size=0.2, random_state=42)

param_grid = {
    'gradientboostingregressor__n_estimators': [50, 100, 200],           
    'gradientboostingregressor__learning_rate': [0.01, 0.05, 0.1, 0.2],  
    'gradientboostingregressor__max_depth': [3, 4, 5, 6],                
    'gradientboostingregressor__subsample': [0.8, 1.0],                  
    'gradientboostingregressor__min_samples_split': [2, 5, 10]           
}

pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)

print("Best cross-validated score: {:.2f}".format(grid_search.best_score_))
```
![image](https://github.com/user-attachments/assets/965fc016-5250-49cb-9de9-c75ca9bd612f)


##### 2.4 Evaluation
- The models were evaluated using R-squared (R²) for variance explained, Mean Squared Error (MSE) for average squared discrepancies, Mean Absolute Error (MAE) for prediction accuracy, and Root Mean Squared Error (RMSE) for assessing average error magnitude, ensuring comprehensive performance evaluation.
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = grid_search.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R² of this model: {r2:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
```
![image](https://github.com/user-attachments/assets/43855f83-779b-4f74-906b-be5d0be08720)

### Results
- The optimized Gradient Boosting model achieved the best performance

| Metric                     | Value |
|----------------------------|-------|
| R² of this model           | 0.80  |
| Mean Squared Error (MSE)   | 0.16  |
| Mean Absolute Error (MAE)  | 0.19  |
| Root Mean Squared Error (RMSE) | 0.40  |

- Key features affecting medical cost predictions included age, bmi, and smoker status, with smoker being a strong indicator of higher charges.

### Conclusion
This project effectively used regression to predict individual medical costs, achieving an R² score of 0.80.

### Next Steps
- **_Explore Additional Algorithms:_** Testing more complex models like neural networks.
- **_Feature Engineering:_** Investigate interactions between features like bmi and age.
- **_Deployment Considerations:_** Consider building a web application to predict medical costs based on input details.
