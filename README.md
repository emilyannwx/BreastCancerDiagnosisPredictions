# BreastCancerDiagnosisPredictions
This project uses Spark MLLib to predict whether a breast tumor is benign (B) or malignant (M) using 20 clinical features from a dataset with 569 samples.

## Algorithms Used
Logistic Regression: Simple, interpretable, and a good baseline for binary classification.
Random Forest: More powerful, handles complex patterns, and often performs better in practice.

## Formatting Data
- Dropped ID column.
- Encoded labels ('M' = 1.0, 'B' = 0.0).
- Selected the first 20 features.
- Converted all data to float.
- Converted data to Spark LabeledPoint format.

## Data Split
- Training: 70%
- Testing: 30%
- Used a fixed seed to keep results consistent.

## Results
### Logistic Regression
Accuracy:  0.9128 (91.28%)
Precision: 0.8548
Recall:    0.8983
F1 Score:  0.8760


### Random Forest
Accuracy:  0.9128 (91.28%)
Precision: 0.8235
Recall:    0.9492
F1 Score:  0.8819

## Evaluation Metrics
Accuracy: Percent of correct predictions
Precision: Correct malignant predictions
Recall: Detected actual malignant cases
F1 Score: Balance between precision and recall

## Model Comparison
Both models had the same accuracy (91.28%), so they performed equally well overall.
Logistic Regression had higher precision, meaning it made fewer false alarms.
Random Forest had higher recall and F1 score, meaning it caught more actual cancer cases and had a better balance overall.

## Conclusion
Random Forest is the better choice for cancer diagnosis because it’s better at detecting true cases and reducing missed diagnoses.










