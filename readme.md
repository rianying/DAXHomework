# PT.DAX Indonesia Candidate Assesment (Math Score Prediction Model)

## Introduction
This project aims to develop a model to predict students Math scores based on their scores in Physics, Science, and Statistics using a dataset of 466 entries.

## Exploratory Data Analysis (EDA) and Findings

### Step 1: Analyze and Visualize Sample Data

```python
data = pd.read_csv('Data.csv')
data.info()
data.head()
data.describe()
```

After we load the data and perform an initial examination, several patterns, correlations, and anomalies were observed that were critical to understand the dataset and to guide model development.

### Data Overview

1. **Data Structure:**
   - The dataset contains 466 entries and 4 variables: "Physics", "Science", "Statistics", and "Math".
   - All variables are integers and there are no missing values, ensuring a smooth modeling process without the need for much transformation.

![Info](https://i.imgur.com/tjndFFO.png)
![Head](https://i.imgur.com/tLjUQHN.png)

2. **Descriptive Statistics:**
   The variables has the following properties:
   - **Physics:** Mean: 71.74, Std Dev: 16.03, Min: 0, Max: 99
   - **Science:** Mean: 72.81, Std Dev: 14.27, Min: 0, Max: 99
   - **Statistics:** Mean: 73.68, Std Dev: 12.18, Min: 22, Max: 100
   - **Math:** Mean: 74.54, Std Dev: 11.42, Min: 22, Max: 99
![Statistic](https://i.imgur.com/r7ASxh5.jpg)

### Visualization Findings
```python
plt.figure(figsize=(10, 8))
sns.pairplot(data, kind="scatter")
plt.suptitle('Pairwise Relationships between Variables', y=1.02)
plt.show()
```
![Plot](https://i.imgur.com/VWJGX1L.png)
An exploration of relationships between the variables via scatter plots shows several insights:

- **Positive Correlation:** There's a positive correlation among all subject scores. This implies a general trend where students scoring high in one subject are much likely to score high in others.
- **Score Distribution:** Most students have scores clustered between 60 and 80 across all subjects, which might indicate a general proficiency or grading trend among the students.
- **Potential Outliers:** Some zeros in "Physics" and "Science" was identified as potential outliers. These might indicate either data entry errors or some occurences where students did not participate in the test.

These findings from the EDA were important in the data preprocessing and model development stages, guiding decisions such as outlier handling and model selection.

## Model Development

### 1. Why Choose Gradient Boosting?

#### Background
Gradient Boosting was chosen after trials and errors using other models (Linear Regression, Random Forest, Polynomial Regression). It is having suboptimal predictive performance, suggesting that the relationship between the predictor variables (Physics, Science, and Statistics scores) and the target variable (Math scores) might be non-linear and complex.

There are few reasoning behind why I chose the Gradient Boosting:

- **Versatility:** It can capture complex non-linear relationships and interactions between features, making it suitable for this problem where linear models performed poorly.
- **Ensemble Learning:** It make use of ensemble learning, combining multiple weak learners (decision trees) to create a strong predictive model, often providing higher accuracy than individual models.
- **Handling Overfitting:** Gradient Boosting has mechanisms to prevent overfitting through regularization and ensemble learning, making it capable of performing well on various kinds of data.

### 2. Model Development Process

#### a. Exploratory Data Analysis (EDA)
Initial visualization and statistical analysis were performed to understand data characteristics and identify potential outliers.

#### b. Data Preprocessing
```python
# Preprocess data: Remove potential outliers
filtered_data = data[data['Math'] >= 10]
```
Potential outliers, specifically scores below 10, were removed to refine the dataset for model training.

#### c. Model Building
##### i. Initialize Model
```python
# Initialize a Gradient Boosting model
gb_model_base = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42))
```
Gradient Boosting was initialized, utilizing a pipeline to standardize the data and then apply the Gradient Boosting Regressor.

##### ii. Hyperparameter Tuning
```python
# Simplified parameter grid
simple_param_grid = {
    'gradientboostingregressor__n_estimators': [50, 100],
    'gradientboostingregressor__learning_rate': [0.01, 0.1],
    'gradientboostingregressor__max_depth': [3, 5],
    'gradientboostingregressor__min_samples_split': [2, 4],
    'gradientboostingregressor__min_samples_leaf': [1, 3]
}

# Initialize Grid Search with simplified grid
simple_grid_search = GridSearchCV(gb_model_base, simple_param_grid, 
                                  cv=5, scoring='r2', n_jobs=-1)
```
Hyperparameter tuning was conducted using a grid search strategy to identify optimal hyperparameters for the Gradient Boosting model.

##### iii. Model Training
```python
# Fit model
simple_grid_search.fit(X_filtered, y_filtered)
```
The model was trained using the dataset, fitting it to the Physics, Science, and Statistics scores to predict Math scores.

#### d. Model Evaluation
```python
# Visualizing actual vs. predicted scores and residuals
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatter

plot(x='Actual', y='Predicted', data=comparison_gb_best)
plt.plot([20, 100], [20, 100], color='red', linestyle='--')
plt.title('Actual vs. Predicted Math Scores (Gradient Boosting)')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')

plt.subplot(1, 2, 2)
sns.histplot(comparison_gb_best['Residuals'], bins=20, kde=True)
plt.title('Distribution of Residuals (Gradient Boosting)')
plt.xlabel('Residuals')

plt.tight_layout()
plt.show()
```
Model evaluation involved assessing how well the model predicted Math scores, using metrics like Mean Absolute Error (MAE) and R^2 Score, and visualizations like Actual vs. Predicted scatter plots and Residual histograms.

![Evaluation](https://i.imgur.com/yzEojrC.png)

#### e. Prediction and Export
```python
# Generate predictions using the best model
best_gb_predictions = simple_grid_search.predict(X_filtered)

# Create a DataFrame to compare actual vs. predicted values
comparison_gb_best = pd.DataFrame({'Actual': y_filtered, 'Predicted': best_gb_predictions})

# Export the predictions to a CSV file
comparison_gb_best.to_csv('/path_to_save_location/predicted_math_scores.csv', index=False)
```
Predictions were generated using the best model from the grid search and exported to a CSV file for further use or analysis.

#### f. Prediction Result
```python
result = pd.read_csv('predicted_math_scores.csv')
print(result.head(10))
```
![Result](https://i.imgur.com/AKxWhts.png)

## Conclusion
The project provided valuable insights into the relationships between scores in Physics, Science, Statistics, and Math. 


Although the Gradient Boosting model provided the best results among those tested, prediction performance was limited, indicating that scores in the three predictor subjects do not strongly determine Math scores. Future work may explore additional data and modeling strategies to enhance predictive performance.
