import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sn

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# reading the dataset
# parameters - path, name of the sheet, not-available value identifier, label row
test_dataset = pd.read_excel(".\\IS4003_SCS4104_CS4104_dataset.xlsx", "Testing Dataset", na_values=["?"], header=0, )
train_dataset = pd.read_excel(".\\IS4003_SCS4104_CS4104_dataset.xlsx", "Training Dataset", na_values=["?"], header=0)


# setting pandas options (purely for convenience)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# setting seed value
tf.random.set_seed(123)


# function to cleanup the dataset
def cleanup_dataset(dataset, name):
    # removing the ID column
    dataset.pop("ID")

    print("Sum of missing values per column in ", name, " :\n", dataset.isnull().sum(), sep="")
    print("Total rows in", name, ":", len(dataset))
    print("Duplicate rows in", name, ":", dataset.duplicated(subset=None, keep=False).sum())

    # removing duplicate rows
    dataset.drop_duplicates(subset=None, keep="first", inplace=True)

    print("# rows after removing duplicates : ", dataset.shape[0])

    # Transform Gender column to a numerical representation
    dataset["Gender"].replace({"Male": 1, "Female": 0}, inplace=True)

    # transform Class column to a numerical representation
    dataset["Class"].replace({"Yes": 1, "No": 0}, inplace=True)

    # Fill the missing values with the median of each respective column
    for key in dataset.columns[dataset.isnull().any()]:
        median = dataset[key].median()
        dataset[key].fillna(median, inplace=True)

    return dataset


# feed the 2 dataset to the cleanup function
train_dataset = cleanup_dataset(train_dataset, "Training dataset")
test_dataset = cleanup_dataset(test_dataset, "Testing dataset")

# separating the Class column from the dataset
train_class = train_dataset.pop("Class")
test_class = test_dataset.pop("Class")

# will contain all the feature_columns that are used in tensorflow
feature_columns = []

# categorical columns in the dataset
categorical = ["Gender"]

# one-hot encoding categorical features and adding them to feature_columns array
for feature in categorical:
    vocabulary = train_dataset[feature].unique()
    feature_columns.append(
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature,
                                                                                                     vocabulary)))

# numerical columns in the dataset
numerical = ["Age", "TB", "DB", "ALK", "SGPT", "SGOT", "TP", "ALB", "AG_Ratio"]

# adding numerical features to feature_column array
for feature in numerical:
    feature_columns.append(tf.feature_column.numeric_column(feature, dtype=tf.float32))


# making the input function
def make_input_fn(ds, cls, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(ds), cls))

        if shuffle:
            dataset = dataset.shuffle(len(train_class))

        # cycle through many times as needed for training
        dataset = dataset.repeat(n_epochs)

        # using the entire training data for the batch
        dataset = dataset.batch(len(train_class))

        return dataset

    return input_fn


# training and testing input functions
train_input_fn = make_input_fn(train_dataset, train_class)
test_input_fn = make_input_fn(test_dataset, test_class, shuffle=False, n_epochs=1)

# using 1 batch per layer
n_batches = 1
model = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=n_batches, learning_rate=0.2)

# training the model
model.train(train_input_fn, max_steps=200)

# get model prediction for evaluation and visualisation
predictions = list(model.predict(test_input_fn))
y_pred = []
for pred in predictions:
    y_pred.append(pred["class_ids"][0])

# plot probabilities in a graph
probabilities = pd.Series([pred["probabilities"][0] for pred in predictions])
probabilities.plot(kind="hist", bins=100, title="Predicted probabilities")
plt.show()

# construct the confusion matrix
CM = confusion_matrix(test_class, y_pred)
confusion = metrics.confusion_matrix(test_class, y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

# calculate the metrics
accuracy = (TP + TN) / float(TP + TN + FP + FN)
classification_error = (FP + FN) / float(TP + TN + FP + FN)
sensitivity = TP / float(FN + TP)
specificity = TN / (TN + FP)
precision = TP / float(TP + FP)

print("Accuracy : {:.4f}".format(accuracy))
print("Precision : {:.4f}".format(precision))
print("Sensitivity : {:.4f}".format(sensitivity))
print("Specificity : {:.4f}".format(specificity))
print("Error rate : {:.4f}".format(classification_error))

print("False Positives : ", FP)
print("False Negatives : ", FN)
print("True Positives : ", TP)
print("True Negatives : ", TN)


# create dataframe for confusion matrix
confusion_matrix_data = pd.DataFrame(CM, columns=np.unique(test_class), index=np.unique(test_class))
confusion_matrix_data = confusion_matrix_data.rename(columns={1: "Positive", 0: "Negative"},
                                                     index={1: "Positive", 0: "Negative"})


# define axis names
confusion_matrix_data.index.name = "Actual"
confusion_matrix_data.columns.name = "Predicted"

# plotting confusion matrix
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4)
fig = sn.heatmap(confusion_matrix_data, annot=True, annot_kws={"size": 22}, fmt="g")
fig.set_title("Confusion Matrix")
plt.show()

