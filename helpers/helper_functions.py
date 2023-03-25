# Data Visualzization
import tensorflow as tf
import matplotlib.pyplot as plt


def loss_curve_plot(df):
    """ Dataframe (df) is history of the fit of the NN model
    The df consists of train and validation fit data
    """
    history = df.history
    val_accuracy = history["val_accuracy"]
    val_loss = history["val_loss"]
    train_accuracy = history["accuracy"]
    train_loss = history["loss"]

    """Accuracy Plot"""
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.title("Accuracy Curves")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    """Loss Plot"""
    plt.plot(train_loss, label="Train loss")
    plt.plot(val_loss, label="Validation loss")
    plt.title("Loss Curves")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


import seaborn as sns
from sklearn import metrics


def confusion_matrix_plot(y_true, y_pred, size=16):
    """"Confusion Matrix for true values and predicted values"""
    cm = metrics.confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", annot_kws={"size": size})


def supervised_metrics(y_true, y_pred):
    """Meterics for a Supervised Learning model:
    TP -> True postivies
    TN -> True negatives
    FP -> False postivies
    FN -> False negatives"""
    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

#Model Helper Functions
import datetime
def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.
  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"
  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

#Unsupervised Clustering Metrics
def supervised_metrics(y_pred, y_true=None, x=None):
  """Meterics for a Unsupervised Learning model
  TP -> True postivies
  TN -> True negatives
  FP -> False postivies
  FN -> False negatives"""
  if(x):
    print("...........Score with respect to X............")
    print(f"Silohouette Score: {metrics.silhouette_score(x,y_pred)}")
    print(f"Calinski Harabasz Score: {metrics.calinski_harabasz_score(x,y_pred)}")
    print(f"Davies Bouldin Score: {metrics.davies_bouldin_score(x,y_pred)}")
  if(y_true):
    print("...........Score with respect to True Labels............")
    print(f"Adjusted Rand Score: {metrics.adjusted_rand_score(y_true, y_pred)}")
    print(f"Completeness Score: {metrics.completeness_score(y_true, y_pred)}")
    print(f"Fowlkes Mallows Score: {metrics.fowlkes_mallows_score(y_true, y_pred)}")
    print(f"Homogeneity Score: {metrics.homogeneity_score(y_true, y_pred)}")

import os
import random
import matplotlib.image as mpimg    

#Image Pre-Processing
def pre_process_image(path, image_shape, channels=3, norm_factor=255.):
    '''Pre-Processing the Image before sending it to the model'''
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=channels)
    img = tf.image.resize(img, size = (image_shape, image_shape))
    img = tf.expand_dims(img, axis=0)
    img = img/norm_factor
    return img

def view_random_image(root_path,folder,class_folder):
    path=root_path+'/'+folder+'/'+class_folder
    rand=random.choice(os.listdir(path))
    random_image=mpimg.imread(path+'/'+rand)
    plt.imshow(random_image)
    plt.title("File Name:" + rand)

def custom_rounder(x,threshold):
  if(x<threshold):
    return 0
  else:
    return 1

def random_tester(root_path, classes, model, class_type="binary"):
    '''Random Class Folder Selection'''
    path=root_path
    class_folder=random.choice(os.listdir(path))

    '''Random File Selection'''
    folder_path='/content/drive/MyDrive/IntelliDoc/chest_xray/test/'+class_folder+'/'
    rand=random.choice(os.listdir(folder_path))
    file_path=folder_path+'/'+rand
    random_image=mpimg.imread(file_path)

    '''Prediction'''
    predicted_value=model.predict(pre_process_image(file_path)) 
    if(class_type=="binary"):
      predicted_label=classes[custom_rounder(predicted_value)]
    else:
      predicted_label=classes[tf.math.round(predicted_value)]

    '''Visualize'''
    plt.imshow(random_image)
    plt.title("Prediction:" + predicted_label +"\n" +"True class: "+ class_folder)
    plt.show()
