# Loan Approval System

This project is a machine learning pipeline for predicting loan approval based on applicant data. It includes data preprocessing, binary classification ,exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Project Structure
- `loan_Approval.py`: Main Python script containing the ML workflow.
- `loan_approval_data.csv`: Dataset used for training and evaluation.
- `flow.drawio.svg`: Flow diagram illustrating the project pipeline.

## Workflow Overview
1. **Importing Libraries**: pandas, numpy, matplotlib, seaborn, sklearn
2. **Handling Missing Data**: Imputation for numerical and categorical features
3. **EDA (Exploratory Data Analysis)**: Visualizations and summary statistics
4. **Feature Encoding**: Label encoding for categorical variables
5. **Feature Engineering**: Creating or transforming features for better model performance
6. **Correlation Heatmap**: Visualizing feature correlations
7. **Model Training and Feature Selection**: Training models and selecting important features
8. **Model Evaluation**: Assessing model performance using metrics

## How to Run
1. Ensure you have Python 3.x installed.
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Place the dataset (`loan_approval_data.csv`) in the project directory.
4. Run the main script:
   ```bash
   python loan_Approval.py
   ```

## Flow Diagram
See `flow.drawio.svg` for a visual representation of the pipeline.

## License
This project is provided for demonstration, portfolio, and personal development purposes. All rights reserved.
