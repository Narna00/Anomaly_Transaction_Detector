# Anomaly Transaction Detection

This project is focused on building and evaluating machine learning models to detect anomalous (fraudulent) transactions. The full workflow is implemented in a Jupyter Notebook (`anomaly.ipynb`).

## Overview of the Process
The notebook walks through the entire pipeline of anomaly detection in financial transactions:

1. **Data Loading**  
   - Imported the dataset (`fraud.csv`) into a pandas DataFrame.
   - Inspected the structure, data types, and summary statistics to understand the distribution of features.

2. **Exploratory Data Analysis (EDA)**  
   - Visualized class distribution to reveal imbalance between fraudulent and normal transactions.
   - Explored correlations between features using heatmaps and pairplots.
   - Created plots to detect patterns and anomalies in transaction amounts and times.

3. **Data Preprocessing**  
   - Cleaned missing values and irrelevant columns where necessary.
   - Encoded categorical variables and scaled numerical features.
   - Split dataset into training and testing sets to enable proper model evaluation.

4. **Model Training**  
   - Trained multiple models with a focus on tree-based classifiers suitable for imbalanced datasets.
   - Models included:
     - Random Forest Classifier
     - LightGBM Classifier
   - Configured hyperparameters for each model to improve performance.

5. **Evaluation**  
   - Used standard metrics such as Accuracy, F1 Score, and Confusion Matrix to evaluate models.
   - Visualized confusion matrices to see how well each model detected fraud cases.
   - Compared performance across models and highlighted trade-offs (e.g., precision vs recall).

6. **Insights**  
   - Identified which features contributed most to fraud detection.
   - Highlighted the challenge of class imbalance and the importance of F1 score over plain accuracy.

7. **Model Saving (Optional)**  
   - Added the option to save trained models with `joblib` for later use.

---

## Requirements
To run the notebook, install the following packages:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn joblib
```

Python 3.8 or higher is recommended.

---

## How to Run
1. Clone this repository or download the files.
2. Place your dataset file (`fraud.csv`) in the project root directory.
3. Launch Jupyter Notebook or Jupyter Lab:

```bash
jupyter notebook
```

4. Open `anomaly.ipynb` and run the cells sequentially.

---

## Results
- The Random Forest and LightGBM models showed strong performance, with LightGBM generally offering better precision and recall on the fraud class.
- The evaluation section of the notebook contains detailed confusion matrices and metric outputs.
- Results confirmed that accuracy alone can be misleading, making F1 score the more reliable metric for imbalanced fraud detection.

---

## Next Steps
- Experiment with resampling techniques (SMOTE, undersampling) to address class imbalance.
- Add cross-validation for more robust model evaluation.
- Extend to additional models like XGBoost and CatBoost for comparison.
- Deploy the best-performing model as a service for real-time transaction scoring.

---

## License
MIT License (you can replace with your preferred license).

---

## Contact
For questions, improvements, or collaboration, feel free to reach out or open an issue on the repository.

