# Anti-Money Laundering Detection Using Machine Learning

## Project Overview
This project leverages Spark ML to analyze and detect patterns of money laundering in financial transactions. By utilizing a dataset that categorizes transactions into high- and low-illicit categories, the goal is to train a robust machine learning model capable of identifying potential money laundering activities.

## Objectives
- Develop a machine learning model to identify patterns associated with money laundering.
- Utilize Spark ML and Databricks for data preprocessing, model training, and evaluation.
- Implement and compare multiple machine learning algorithms:
  - Logistic Regression
  - LinearSVC
  - Random Forest
  - Gradient Boosted Tree

## Platform Specifications
- **Hadoop Version**: 3.3.3
- **Spark Version**: 3.2.1
- **Cluster Configuration**: 5 nodes, 8 CPU cores, 860.4 GB total memory
- **Dataset**: IBM Transactions for Anti-Money Laundering (2.82 GB in CSV format, available on Kaggle)

## Dataset Specifications
- **Dataset Name**: IBM Transactions for Anti-Money Laundering (AML)
- **File Used**: `HI_Medium_Trans.csv` (2.82 GB)
- **Format**: CSV

## Key Steps
1. **Data Acquisition**: Download the dataset from Kaggle and upload it to Databricks.
2. **Cluster Setup**: Configure a Databricks cluster to support the dataset and algorithms.
3. **Data Processing**:
   - Define schema and load data into a DataFrame.
   - Balance the dataset using undersampling.
   - Convert categorical data to numerical indices for compatibility with ML algorithms.
4. **Model Training**:
   - Split the dataset into training and test sets (70/30 split).
   - Train models using multiple algorithms:
     - Logistic Regression
     - Random Forest Classifier
     - Gradient Boosted Trees
     - Linear Support Vector Classifier
   - Implement both Train-Validation Split and Cross-Validation for optimal performance.
5. **Evaluation Metrics**:
   - Calculate Precision, Recall, and AUC to assess model performance.
   - Display feature importance for interpretability, especially in the Gradient Boosted Tree model, which achieved the highest performance.

## Running the Project
1. **Clone the Repository**: Clone this repository to access all project files.
2. **Dataset Upload**: Download the AML dataset from Kaggle and upload it to Databricks.
3. **Run Code in Databricks**:
   - Set up a Databricks cluster with specified Spark and Hadoop configurations.
   - Upload and execute the provided Databricks notebook for data preprocessing, model training, and evaluation.
4. **Run with Spark-Submit**: 
   For local or non-Databricks Spark clusters, use the `spark-submit` command to run the code:
   ```bash
   spark-submit AML_LogisticRegression_TVFinal_UnderSample.py

## Code Files and Execution Details
All code files are available in this repository. For a step-by-step breakdown of code and model execution, please refer to the Project Tutorial.

## Results
The Gradient Boosted Tree model demonstrated the best performance, with high Recall and AUC scores, proving valuable in enhancing detection capabilities for AML systems. Feature importance analysis highlights key features, aiding in interpretability and real-world applications.

## License
This project is licensed under the MIT License.

## Contact Information
For questions or feedback, please contact:
- **Snehil Sarkar**: [snehil.sarkar100@gmail.com](mailto:snehil.sarkar100@gmail.com)
