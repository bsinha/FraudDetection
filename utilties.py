from sklearn.metrics import confusion_matrix, roc_curve, auc
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve


def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plots a confusion matrix based on true and predicted labels.

    Parameters:
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    labels : list of str
        List of class labels.
    """
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualize confusion matrix using heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


# The ROC curve plots the True Positive Rate (TPR) on the y-axis against the False Positive Rate (FPR) on the x-axis as the discrimination threshold is varied.
#  - It helps visualize the model's ability to distinguish between fraudulent and legitimate transactions.
#  - A model with a higher AUC (Area Under the Curve) is generally better at discrimination.

def plot_roc_curve(y_true, y_pred):
  """
  Plots the ROC curve for a model.

  Parameters:
    y_true : array-like of shape (n_samples,)
      True labels.
    y_pred : array-like of shape (n_samples,)
      Predicted probabilities.
  """
  fpr, tpr, thresholds = roc_curve(y_true, y_pred)
  roc_auc = auc(fpr, tpr)

  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], 'k--', label='No Discrimination')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate (FPR)')
  plt.ylabel('True Positive Rate (TPR)')
  plt.title('ROC Curve')
  plt.legend(loc="lower right")
  plt.show()



# Precision-Recall Curve(Use for Imbalanced DataSet):

# - The precision-recall curve plots Precision (positive predictive value) on the y-axis against Recall (true positive rate) on the x-axis as the classification threshold is varied.
# - This is useful when dealing with imbalanced datasets, where positive cases (fraudulent transactions) might be rare.
# - A model with a curve that stays closer to the top-left corner indicates a better balance between precision and recall.
def plot_precision_recall_curve(y_true, y_pred):
  """
  Plots the precision-recall curve for a model.

  Parameters:
    y_true : array-like of shape (n_samples,)
      True labels.
    y_pred : array-like of shape (n_samples,)
      Predicted labels.
  """
  precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

  plt.figure(figsize=(8, 6))
  plt.plot(recall, precision, label='Precision-Recall Curve')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall Curve')
  plt.legend(loc="lower left")
  plt.show()


# Distribution Plots:

# - Create histograms or kernel density estimation (KDE) plots to visualize the distribution of features or predicted probabilities for both fraudulent and legitimate transactions.
# - This can help identify potential patterns or outliers that might be related to fraudulent activity.

def plot_distribution(data, feature_name, class_label="class_label", kind="kde"):
  """
  Plots the distribution of a feature for different classes.

  Parameters:
    data : pandas DataFrame
      DataFrame containing the data.
    feature_name : str
      Name of the feature to plot.
    class_label : str, optional
      Name of the class label column (default: "class_label").
    kind : str, optional
      Plot kind (e.g., "hist" for histogram, "kde" for kernel density estimation).
  """
  sns.displot(data=data, x=feature_name, hue=class_label, kind=kind)
  plt.title(f'Distribution of {feature_name} by {class_label}')
  plt.show()


def plot_predicted_probability_distribution(y_pred, bins=10):
  """
  Plots the distribution of predicted probabilities.

  Parameters:
    y_pred : array-like
      Array of predicted probabilities.
    bins : int, optional
      Number of bins for the histogram (default: 10).
  """
  sns.displot(y_pred, bins=bins, kde=True)
  plt.title('Distribution of Predicted Probabilities')
  plt.show()



def show_result(y_test, y_pred, labels):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy::  {accuracy:.2f}")
    
    precision = precision_score(y_test, y_pred)
    print(f"Precision Score::  {precision:.2f}")
    
    recall = recall_score(y_test, y_pred)
    print(f"Recall Score::  {recall:.2f}")
    
   # f1score = f1_score(precision, recall)
    #print(f"F1-Score::  {f1score:.2f}")
    
    confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, labels)
    plot_roc_curve(y_test, y_pred)
    plot_predicted_probability_distribution(y_pred)
    
    return {"accuracy" : accuracy, "precision" : precision, "recall" : recall}
