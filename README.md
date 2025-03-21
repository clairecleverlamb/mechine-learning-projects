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


-------------------------------------------------------------------------------------------------------------------------


# 2. Unsupervised Learning: Document Clustering and Topic Modeling

This project leverages unsupervised learning techniques to cluster and analyze a collection of text documents, specifically focusing on product reviews. Using K-means clustering and Latent Dirichlet Allocation (LDA), we group similar documents into clusters and uncover latent topics within the corpus. The project is implemented in Python and designed to run in a Google Colab environment.

## Dataset
The dataset consists of 960,056 text documents (product reviews) loaded from a tab-separated file named 'data.tsv'. After removing missing values from the 'review_body' column, we select the first 1,000 documents for analysis to ensure computational feasibility in this exploratory study. The dataset includes fields such as marketplace, customer_id, review_id, product_id, and review_body, but only the review_body text is used for clustering and topic modeling.

## Setup
This project is optimized for Google Colab. Follow these steps to set up and run the project:

### Authenticate with Google Drive:
- The dataset is accessed from Google Drive using PyDrive.
- Replace the file ID '192JMR7SIqoa14vrs7Z9BXO3iK89pimJL' in the code with the ID of your own data.tsv file if necessary.

### Install Dependencies:
The project requires the following Python libraries:
- numpy
- pandas
- nltk
- scikit-learn
- matplotlib
- PyDrive

Install them in a Colab cell with:
```bash
!pip install -U -q PyDrive numpy pandas nltk scikit-learn matplotlib
```

### Download NLTK Data:
The code uses NLTK's 'punkt' (for tokenization) and 'stopwords' (for stop word removal). These are downloaded automatically via:
```python
nltk.download('punkt')
nltk.download('stopwords')
```

### Upload the Dataset:
Ensure data.tsv is uploaded to your Google Drive and accessible with the correct file ID.

## Project Structure
The project is organized into five main parts, each representing a key step in the analysis pipeline:

### Part 0: Setup Google Drive Environment
Authenticate with Google Drive and download data.tsv using PyDrive.
Example:
```python
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
file = drive.CreateFile({'id': '192JMR7SIqoa14vrs7Z9BXO3iK89pimJL'})
file.GetContentFile('data.tsv')
```

### Part 1: Load Data
Load the dataset into a Pandas DataFrame from data.tsv. Clean the data by removing rows with missing review_body values and select the first 1,000 reviews.
Example:
```python
df = pd.read_csv('data.tsv', sep='\t', error_bad_lines=False)
df.dropna(subset=['review_body'], inplace=True)
data = df.loc[:999, 'review_body'].tolist()
```

### Part 2: Tokenizing and Stemming
- Tokenize the text into words using NLTK's punkt tokenizer.
- Apply stemming to reduce words to their root form (e.g., "running" → "run").
- Remove stop words (e.g., "the", "and") to focus on meaningful content.

### Part 3: TF-IDF
Convert the preprocessed text into a TF-IDF (Term Frequency-Inverse Document Frequency) matrix using TfidfVectorizer from scikit-learn. TF-IDF weights words based on their importance in the documents, reducing the influence of common terms.
Example:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_model = TfidfVectorizer()
tfidf_matrix = tfidf_model.fit_transform(data)
```

### Part 4: K-means Clustering
Apply K-means clustering to group similar documents based on the TF-IDF matrix. Determine the optimal number of clusters (e.g., using the elbow method) and visualize results (e.g., using PCA for dimensionality reduction).

### Part 5: Topic Modeling - Latent Dirichlet Allocation
Use Latent Dirichlet Allocation (LDA) to identify underlying topics in the document corpus. Fit an LDA model with 5 topics to the TF-IDF matrix and extract topic-word distributions.
Example:
```python
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=5)
lda_output = lda.fit_transform(tfidf_matrix)
```

Analyze the dominant topic for each document and the top words associated with each topic.

Example output:
- `lda_output (1000, 5)`: Document-topic matrix showing topic probabilities for each document.
- `lda.components_ (5, 239)`: Topic-word matrix showing word weights for each topic.

## Results
### Clustering:
- While not fully implemented in the provided code, K-means clustering would group the 1,000 reviews into distinct clusters based on textual similarity, revealing patterns such as positive vs. negative sentiment or product-specific feedback.

### Topic Modeling:
- The LDA model identified 5 latent topics within the reviews.
- Example analysis:
  - A `df_document_topic` DataFrame shows the topic distribution for each document, with the dominant topic assigned.
  - Top words per topic (extracted via `print_topic_words`) provide interpretable themes, such as product quality, customer satisfaction, or specific features (e.g., "watch", "love", "great" might dominate one topic).
  - Sample topic distribution: Topic 4 was dominant in many documents (e.g., 49.99% in Doc0), indicating a prevalent theme in the reviews.

## Usage
### Run the Notebook:
- Open the project in Google Colab.
- Execute the cells sequentially to load data, preprocess it, and perform clustering and topic modeling.

### Modify Parameters:
- Adjust the number of clusters in K-means or topics in LDA (e.g., `n_components=5`) to explore different groupings.
- Experiment with preprocessing (e.g., additional stop words or stemming options) to refine results.

### Interpret Results:
- Use `df_document_topic` to identify dominant topics per document.
- Analyze `df_topic_words` or the output of `print_topic_words` to understand the themes uncovered by LDA.

## Insights
- The LDA model successfully uncovered coherent topics within the first 1,000 reviews, with topic distributions varying across documents (e.g., some documents are evenly split across topics, while others strongly align with one).
- The project demonstrates the power of unsupervised learning for exploratory text analysis, offering insights into customer opinions without predefined labels.
- Potential topics might include "positive feedback on watch design" or "complaints about durability," though specific interpretations depend on the top words extracted.



