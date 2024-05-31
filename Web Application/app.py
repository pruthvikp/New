from flask import Flask, request, render_template
import pandas as pd
import string
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
with open('yelp_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to preprocess text
def remove_punc_stopword(text):
    remove_punc = [word for word in text if word not in string.punctuation]
    remove_punc = ''.join(remove_punc)
    return [word.lower() for word in remove_punc.split() if word.lower() not in stopwords.words('english')]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        prediction = model.predict(data)
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
