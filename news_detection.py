import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def plot(X, Y, save_path):
  plt.plot(X, Y)
  plt.xlabel('# of tests')
  plt.ylabel('accuracies')
  plt.savefig(save_path)

def calculateAccuracy(predictions, labels):
  # Calculates the accuracy of the neural network by
  # comparing the predictions calculated with the provided labels
  matches = 0
  mis_prediction = [0, 0] # Real:Fake News ratio

  for i in range(len(predictions)):
    if (predictions[i] == labels[i]):
      matches += 1
  # Used for debugging to find authenticity of mis-predicted articles
  #   else:
  #     if (labels[i] == True):
  #       mis_prediction[0] += 1
  #     else:
  #       mis_prediction[1] += 1
  
  # print('True Articles: ', mis_prediction[0], 'Fake Articles: ', mis_prediction[1])
  return matches / len(predictions)

def compare_prediction(prediction_A, prediction_B):
  # Compares and returns the overall prediction of all predictions in the provided datasets
  merged_predictions = []
  for i in range(len(prediction_A)):
    merged_predictions.append(prediction_A[i] * prediction_B[i])

  # Returns the overall predictions
  return merged_predictions

def run_prediction(X, Y=None, model=None, vectorizer_X=None):
  # Generates a numerical representation of the dataset to be used for the
  # Logistic Regression model 
  if (vectorizer_X is None):
    vectorizer_X = TfidfVectorizer()
    vectorizer_X.fit(X)

  X = vectorizer_X.transform(X)

  # Used to obtain the most probable words used in fake articles
  # fake_word_mapping = X.ceil()
  # print(vectorizer_X.get_feature_names()['id of top five terms'])

  # Performs fitting of the Dataset and labels if a model is not provided
  if (model is None):
    LogisticModel = LogisticRegression()
    LogisticModel.fit(X, Y)
  else:
    LogisticModel = model
  
  # Performs prediction using Logistic Regression on the provided dataset
  prediction = LogisticModel.predict(X)
  
  # Returns an array of all predictions for the dataset, as well as the updated neural
  # network model and vectorized component
  return prediction, LogisticModel, vectorizer_X

def predict_single_article(article, title_model, text_model, title_vector, text_vector):
  # Separates news article into a title and text components
  title = article['title'].values
  text = article['text'].values

  # Runs the prediction on the title and text components using the provided neural network models and vectorized values
  title_prediction_response, title_model, title_vector = run_prediction([title[0]], None, title_model, title_vector)
  text_prediction_response, text_model, text_vector = run_prediction([text[0]], None, text_model, text_vector)

  # Calculates the overall prediction for the current news articles and returns the prediction, as well as the updated
  # neural network models and vectorized components
  prediction = title_prediction_response[0] * text_prediction_response[0]

  return prediction, title_model, text_model, title_vector, text_vector


def main():
  # Data Pre=processing
  true_news = pd.read_csv('./Dataset/True.csv', sep=',', usecols=[0, 1])
  fake_news = pd.read_csv('./Dataset/Fake.csv', sep=',', usecols=[0, 1])

  # Adding a new column to indicate the validity of each article in the dataframe
  true_news["validity"] = False
  fake_news["validity"] = True

  # Merges both true and fake news datasets into a single dataset
  # Shuffles all articles to randomize the validity of each article
  merged_dataset = pd.concat([true_news, fake_news])
  merged_dataset = shuffle(merged_dataset)

  # Create Randomized Train/Text Datasets
  # Utilizes a 30% Training/70% Test split
  train_dataset = merged_dataset.head(int(merged_dataset.shape[0] * 0.7))
  test_dataset = merged_dataset.tail(int(merged_dataset.shape[0] * 0.3) + 1)

  # Running the Model on the Training Dataset
  train_X = train_dataset.iloc[:, 0:2]
  train_Y = train_dataset.iloc[:, 2]

  title_prediction, train_title_model, train_title_vector = run_prediction(train_X['title'].values, train_Y.values)
  text_prediction, train_text_model, test_title_vector = run_prediction(train_X['text'].values, train_Y.values)

  combined_prediction = compare_prediction(title_prediction, text_prediction)

  calculated_accuracy = calculateAccuracy(combined_prediction, train_Y.values)
  
  print("Logistic Regression Training Model returned an accuracy of: ", calculated_accuracy)
  
  # Running the Test Dataset
  test_X = test_dataset.iloc[:, 0:2]
  test_Y = test_dataset.iloc[:, 2]

  accuracies = np.array([])

  # # Code used to plot the accuracies after each test
  # for i in range(int(len(test_X['title'].values))):
  #   # Returns the calculated predictions for the test dataset for both the title and the text values
  #   title_prediction, train_title_model, train_title_vector = run_prediction(test_X['title'].values[:i + 1], None, train_title_model, train_title_vector)
  #   text_prediction, train_text_model, test_title_vector = run_prediction(test_X['text'].values[:i + 1], None, train_text_model, test_title_vector)

  #   # Combines the above predictions and returns the overall accuracy of the entire dataset
  #   combined_prediction = compare_prediction(title_prediction, text_prediction)
  #   calculated_accuracy = calculateAccuracy(combined_prediction, test_Y.values)

  #   accuracies = np.append(accuracies, calculated_accuracy)
  
  # # Plots all accuracies calculated during each iteration of the test dataset
  # plot(range(int(len(test_X['title'].values))), accuracies, './test_accuracies')
  
  # Returns the calculated predictions for the test dataset for both the title and the text values
  title_prediction, train_title_model, train_title_vector = run_prediction(test_X['title'].values, None, train_title_model, train_title_vector)
  text_prediction, train_text_model, test_title_vector = run_prediction(test_X['text'].values, None, train_text_model, test_title_vector)

  # Combines the above predictions and returns the overall accuracy of the entire dataset
  combined_prediction = compare_prediction(title_prediction, text_prediction)

  # Calculates the bias and variance used for bias-variance trade off
  variance = np.var(combined_prediction)
  sse = np.mean((np.mean(combined_prediction) - test_Y.values)** 2)
  bias = sse - variance
  calculated_accuracy = calculateAccuracy(combined_prediction, test_Y.values)

  accuracies = np.append(accuracies, calculated_accuracy)
  
  print("Logistic Regression Test Model returned an accuracy of: ", calculated_accuracy)
  print("Logistic Regression Test Model returned an bias of: ", bias)
  print("Logistic Regression Test Model returned an variance of: ", variance)

  # Tests individual news articles from a random fake and real news article
  # Prints the respective prediction, and updates the Neural Network Model with the newly added article's data
  single_fake_news_test_article = fake_news.head(1)
  test_prediction_fake, train_title_model, train_text_model, train_title_vector, test_title_vector = predict_single_article(single_fake_news_test_article, train_title_model, train_text_model, train_title_vector, test_title_vector)

  print("Fake Article Test returns result of:", 'Fake' if test_prediction_fake else 'Real')

  single_real_news_test_article = true_news.head(1)
  test_prediction_real, train_title_model, train_text_model, train_title_vector, test_title_vector = predict_single_article(single_real_news_test_article, train_title_model, train_text_model, train_title_vector, test_title_vector)

  print("Real Article Test returns result of:",  'Fake' if test_prediction_real else 'Real')

  return train_title_model, train_text_model, train_title_vector, test_title_vector

if __name__ == '__main__':
  main()
