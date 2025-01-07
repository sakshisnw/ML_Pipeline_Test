# ML_Pipeline_Test: The Magic Behind the Machine Learning Process 

## **🚀 Purpose**
Welcome to the world of machine learning! This project takes you through a powerful pipeline that processes datasets, handles missing data, reduces features, and optimizes models with hyperparameter tuning—all in one seamless flow. The beauty of this pipeline? You control everything through a simple JSON configuration file. Whether you're working on **regression** or **classification**, this project adapts to your needs and ensures you get the best model for your task.

## **🔑 Key Features**
Here’s what makes this pipeline shine:
1. **Smart Model Selection:**  
   Automatically chooses the right model for your task (regression or classification). From Linear Regression to Random Forests, only the best-suited models are trained.

2. **Advanced Feature Reduction:**  
   - **PCA**: Reduces dimensions for better efficiency.  
   - **Correlation with Target**: Selects features most relevant to your target.  
   - **Tree-Based Importance**: Picks features based on their importance from tree models like Random Forests.

3. **Effortless Hyperparameter Tuning:**  
   Optimizes model performance using GridSearchCV to find the best settings automatically.

4. **Comprehensive Evaluation:**  
   Logs key metrics like accuracy, RMSE, and R² to help you assess model performance.

5. **Modular & Scalable:**  
   The entire workflow, from preprocessing to evaluation, is neatly packed into reusable sklearn Pipelines for cleaner, scalable code.


## **🛠️ How to Run the Project**

### **📦 Dependencies**
Before diving in, ensure you’ve got these Python packages installed:
- `pandas` – For smooth data manipulation.
- `scikit-learn` – To build and evaluate machine learning models.
- `numpy` – Essential for numerical operations.

You can install everything with a single command:

```bash
pip install pandas scikit-learn numpy
```

### **🏃‍♀️ Running the Code**
Once you've installed the dependencies, fire up the project by running:

```bash
python main.py
```

This command will kickstart the pipeline, pulling the configuration from the `algoparams_from_ui.json` file and loading your dataset (like `iris.csv`) for training and evaluation. Prepare for a model-building experience like no other!

## **🔍 Pipeline Workflow**

### **1. Load and Parse Configuration**:
The configuration file `algoparams_from_ui.json` provides the necessary parameters, including:
- **Prediction Type**: Whether the task is regression or classification.
- **Feature Handling**: Details on how to process features (e.g., PCA, correlation-based reduction, or tree-based feature importance).
- **Algorithm Selection**: Specifies which models (e.g., `RandomForestRegressor`, `LinearRegression`, `RandomForestClassifier`) should be used.

### **2. Dataset Handling**:
- **Loading Data**: The dataset (e.g., `iris.csv`) is loaded as a pandas DataFrame.
- **Cleaning Data**: Missing values are handled by dropping rows with null values.

### **3. Feature Preprocessing**:
- **Categorical Variables**: Categorical features are encoded using `LabelEncoder`.
- **Numerical Variables**: Features are scaled using `StandardScaler` to ensure consistency across models.

### **4. Feature Reduction**:
- If specified in the configuration, feature reduction is applied using one of the following methods:
  - **PCA**: Principal Component Analysis reduces the feature space.
  - **Correlation with Target**: Select the top features based on their correlation with the target.
  - **Tree-Based Feature Importance**: Uses feature importance from tree-based models like `RandomForest` to select the best features.

### **5. Model Training and Evaluation**:
- **Train-Test Split**: The data is split into training and testing sets using `train_test_split`.
- **Model Training**: Based on the configuration, models like `LinearRegression`, `RandomForestRegressor`, and `RandomForestClassifier` are trained.
- **GridSearchCV**: Hyperparameter tuning is performed automatically for selected models.
- **Model Evaluation**: Standard metrics like **accuracy**, **RMSE**, and **R²** are logged for the trained models.

### **6. Logging Results**:
The pipeline evaluates each model's performance and logs the results:
- **For Classification**: Accuracy, Precision, Recall, F1-Score.
- **For Regression**: RMSE, R², and other relevant metrics.

---

## **📊 Areas for Improvement and Future Work**

### **1. Model Handling Based on `prediction_type`**:
The pipeline currently allows for both regression and classification models, but it doesn't dynamically check the `prediction_type` when selecting models. Future improvements will include:
- **Automatic Model Selection**: Only train regression models (e.g., `LinearRegression`, `RandomForestRegressor`) for regression tasks and classification models (e.g., `RandomForestClassifier`, `LogisticRegression`) for classification tasks.
  
### **2. Tree-Based Feature Reduction**:
Currently, feature reduction is implemented using **PCA** and **Correlation with Target**. However, **Tree-based feature importance** (using models like Random Forest) could be added to enhance feature selection by identifying important features based on model performance.

### **3. Logging Standard Metrics**:
The pipeline logs predictions but does not log evaluation metrics like **accuracy**, **RMSE**, **R²**, etc. In the next iteration, we will integrate **sklearn.metrics** (e.g., `accuracy_score`, `mean_squared_error`, `r2_score`) to track model performance more comprehensively.

### **4. Fully Integrated Pipelines**:
Currently, we are using **sklearn pipelines** for preprocessing, but we can enhance this further:
- **Modular Pipelines**: Fully integrate data preprocessing, feature handling, feature reduction, and model training into a single pipeline. This would allow for better manageability and reusability.

---

## **💡 Future Enhancements**
We are always improving this pipeline! Future versions will include:
- **Expanded Model Selection**: More models for both regression and classification.
- **Advanced Feature Selection**: Add more advanced techniques for feature selection, including **Tree-based** methods and **Recursive Feature Elimination (RFE)**.
- **Model Comparison**: Automate the comparison of multiple models to select the best-performing one.
- **Scalability**: Optimize the pipeline for larger datasets and more complex tasks.
