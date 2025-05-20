from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.linalg import Vectors
import pandas as pd


#load + prepare data
df = pd.read_csv("project3_data.csv")
df = df.drop(columns=["id"])
df["diagnosis"] = df["diagnosis"].map({"M": 1.0, "B": 0.0})

X = df.iloc[:, 1:21].astype(float).values.tolist()  #20 features
y = df["diagnosis"].astype(float).tolist()


#create spark rdd[labeledpoint] with float data
labeled_points = [
    LabeledPoint(float(label), Vectors.dense([float(f) for f in features]))
    for label, features in zip(y, X)
]

conf = SparkConf().setAppName("CancerDiagnosisFinalFixed").setMaster("local[*]")
sc = SparkContext(conf=conf)

rdd = sc.parallelize(labeled_points)
train_rdd, test_rdd = rdd.randomSplit([0.7, 0.3], seed=42)

#train the models
print("\nTraining Logistic Regression...")
lr_model = LogisticRegressionWithLBFGS.train(train_rdd, iterations=100, numClasses=2)

print("\nTraining Random Forest...")
rf_model = RandomForest.trainClassifier(
    train_rdd,
    numClasses=2,
    categoricalFeaturesInfo={},
    numTrees=10,
    featureSubsetStrategy="auto",
    impurity="gini",
    maxDepth=5,
    maxBins=32
)

#compute predictions on driver to avoid spark context serialization errors
rf_predictions = [(float(p.label), float(rf_model.predict(p.features))) for p in test_rdd.collect()]
lr_predictions = [(float(p.label), float(lr_model.predict(p.features))) for p in test_rdd.collect()]

#create spark rdds again for evaluation
rf_results = sc.parallelize(rf_predictions)
lr_results = sc.parallelize(lr_predictions)

#evaluation
lr_metrics = BinaryClassificationMetrics(lr_results)
lr_accuracy = lr_results.filter(lambda x: x[0] == x[1]).count() / float(lr_results.count())
rf_metrics = BinaryClassificationMetrics(rf_results)
rf_accuracy = rf_results.filter(lambda x: x[0] == x[1]).count() / float(rf_results.count())

def classification_metrics(results, model_name):
    TP = results.filter(lambda x: x[0] == 1.0 and x[1] == 1.0).count()
    TN = results.filter(lambda x: x[0] == 0.0 and x[1] == 0.0).count()
    FP = results.filter(lambda x: x[0] == 0.0 and x[1] == 1.0).count()
    FN = results.filter(lambda x: x[0] == 1.0 and x[1] == 0.0).count()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"\n{model_name} Classification Report")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

classification_metrics(lr_results, "Logistic Regression")
classification_metrics(rf_results, "Random Forest")

sc.stop()
