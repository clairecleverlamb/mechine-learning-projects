# Mechine Learning Projects


# 1. Supervised Learning: Bank Customer Churn Prediction

This project uses supervised machine learning models to identify bank customers who are likely to churn (leave the bank) in the future. Additionally, it analyzes the top factors influencing customer retention to provide actionable insights for the bank. The project leverages a dataset of bank customers and applies data exploration, preprocessing, model training, and evaluation techniques to achieve its objectives.

## Dataset

The dataset, stored in `bank_churn.csv`, contains information about 10,000 bank customers with no missing values. It includes the following features:

### Customer Demographics:
- **CreditScore**: Customer's credit score (integer)
- **Geography**: Customer's location (categorical: France, Spain, Germany)
- **Gender**: Customer's gender (categorical: Male, Female)
- **Age**: Customer's age (integer)

### Account Details:
- **Tenure**: Years as a bank customer (integer)
- **Balance**: Account balance (float)
- **NumOfProducts**: Number of bank products used (integer)
- **HasCrCard**: Has a credit card (binary: 0 or 1)
- **IsActiveMember**: Active member status (binary: 0 or 1)

### Salary:
- **EstimatedSalary**: Estimated annual salary in USD (float)

### Churn Status:
- **Exited**: Whether the customer churned (binary: 0 for stayed, 1 for left; target variable)

Additionally, the dataset includes identifiers (`RowNumber`, `CustomerId`, `Surname`), which are dropped during preprocessing as they are irrelevant to prediction.

## Methodology

The project is divided into four main parts, each corresponding to a key stage in the machine learning pipeline:

### Part 0: Setup Google Drive Environment / Data Collection
**Objective**: Load the dataset into the environment.
**Steps**:
- Authenticate with Google Drive using PyDrive.
- Download `bank_churn.csv` from Google Drive using the file ID: `1szdCZ98EK59cfJ4jG03g1HOv_OhC1oyN`.
- Load the dataset into a Pandas DataFrame for analysis.

### Part 1: Data Exploration
**Objective**: Understand the dataset and explore feature distributions and relationships with churn.
**Steps**:
- **Basic Analysis**: Examine dataset structure (`info()`), unique values (`nunique()`), and check for missing values (none found).
- **Numerical Features**: Use `describe()` to summarize statistics and boxplots (via Seaborn) to visualize distributions of `CreditScore`, `Age`, `Tenure`, `NumOfProducts`, `Balance`, and `EstimatedSalary` split by `Exited`.
- **Categorical Features**: Use countplots to explore relationships between `Geography`, `Gender`, `HasCrCard`, `IsActiveMember`, and `Exited`.

### Part 2: Feature Preprocessing
**Objective**: Prepare the data for model training.
**Steps**:
- **Drop Irrelevant Features**: Remove `RowNumber`, `CustomerId`, `Surname`, and separate `Exited` as the target variable `y`.
- **Split Data**: Divide into training (75%) and testing (25%) sets using stratified sampling to maintain class distribution (`stratify=y`).
- **Encode Categorical Features**:
  - One-hot encoding for `Geography` (creates columns like `Geography_France`, `Geography_Germany`, `Geography_Spain`).
  - Ordinal encoding for `Gender` (e.g., Male=0, Female=1).
- **Standardize Numerical Features**: Apply `StandardScaler` to normalize `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, and `EstimatedSalary` using training set statistics.

### Part 3: Model Training and Results Evaluation
**Objective**: Train models, tune hyperparameters, and evaluate performance.
**Steps**:
- **Model Training**:
  - Train three supervised learning models: Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest.
- **Hyperparameter Tuning (via Grid Search with 5-fold cross-validation)**:
  - Logistic Regression: Tuned penalty (`l1`, `l2`) and `C` (0.01, 0.05, 0.1, 0.2, 1). Best: `penalty=l1, C=1`.
  - KNN: Tuned `n_neighbors` (1, 3, 5, 7, 9). Best: `n_neighbors=9`.
  - Random Forest: Tuned `n_estimators` (60, 80, 100) and `max_depth` (1, 5, 10). Best: `n_estimators=80, max_depth=10`.
- **Evaluation Metrics**:
  - Confusion Matrix: Calculate accuracy, precision, and recall for each model.
  - ROC Curve and AUC: Plot ROC curves and compute AUC for Random Forest and Logistic Regression.
- **Additional Analysis**:
  - Add a new feature `SalaryInRMB` (`EstimatedSalary * 6.4`) to explore its impact.
  - Use L1 and L2 regularization in Logistic Regression to assess feature importance.

## How to Run the Code

This project is implemented in a Jupyter notebook designed to run in Google Colab. Follow these steps:

1. **Clone or Download**:
   - Clone the repository or download the notebook file.
2. **Upload to Colab**:
   - Open Google Colab and upload the notebook.
3. **Access the Dataset**:
   - Ensure `bank_churn.csv` is available in your Google Drive.
   - Update the file ID in the code if necessary (current ID: `1szdCZ98EK59cfJ4jG03g1HOv_OhC1oyN`).
4. **Run the Notebook**:
   - Execute the cells sequentially to perform setup, data exploration, preprocessing, training, and evaluation.

## Requirements

The following Python libraries are required:

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Plotting
- `seaborn`: Enhanced visualizations
- `scikit-learn`: Machine learning models and tools
- `pydrive`: Google Drive integration

Install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn pydrive
```

## Results

### Model Performance:
- **Random Forest**: Best performer with accuracy ~86.36%, precision ~81.58%, recall ~42.63%, AUC = 0.845.
- **KNN**: Accuracy ~84.28%, precision ~72.83%, recall ~36.35%.
- **Logistic Regression**: Accuracy ~80.92%, precision ~59.64%, recall ~19.45%, AUC = 0.772.

### Key Factors Influencing Churn (from Logistic Regression with L1/L2 regularization):
- **Age**: Strong positive influence (older customers more likely to churn).
- **IsActiveMember**: Strong negative influence (active members less likely to churn).
- **Geography_Germany**: Positive influence (customers in Germany more likely to churn).
- **Gender**: Negative influence (females less likely to churn than males).
- **Balance**: Positive influence (higher balances linked to churn).

These insights suggest the bank could focus retention efforts on older customers, those in Germany, and those with higher balances, while encouraging active membership.

