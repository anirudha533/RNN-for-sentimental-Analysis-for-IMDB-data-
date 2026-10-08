Sentiment Analysis on IMDB Reviews using an RNN (PyTorch)

A binary sentiment classifier that predicts whether a movie review is positive or negative. Reviews are cleaned with NLTK, converted to TF-IDF vectors, and fed into a simple Recurrent Neural Network built with PyTorch.

Dataset
IMDB Dataset of 50K Movie Reviews (IMDB Dataset.csv)
50,000 reviews with two columns: review (text) and sentiment (positive / negative)
After dropping duplicates: 49,582 reviews
Download from Kaggle and place the CSV in the project folder.
Project Workflow
Load data with pandas and check for nulls and duplicates
Pre-processing
Convert text to lowercase
Remove URLs
Remove punctuation and special characters
Remove HTML tags
Remove stopwords (NLTK)
Stemming (Porter Stemmer)
Encoding the labels with LabelEncoder (negative = 0, positive = 1)
Vectorization with TfidfVectorizer (top 5,000 features)
Train/test split: 80% train (39,665 samples) / 20% test (9,917 samples), random_state=42
DataLoaders with batch size 64
Model: RNN (PyTorch)
nn.RNN layer, hidden size 128, 1 layer
Fully connected layer (128 → 1)
Sigmoid on the output to get a probability
Training: Adam optimizer, BCE loss, 10 epochs
Evaluation: accuracy on the test set (threshold 0.5)
Tech Stack
Python 3.13
pandas
NLTK
scikit-learn
PyTorch
Jupyter Notebook
