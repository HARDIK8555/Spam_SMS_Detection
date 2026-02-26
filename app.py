from flask import Flask, render_template, request
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# load model
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

ps = PorterStemmer()
app = Flask(__name__)

# preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

@app.route('/')
def home():
    return render_template(
    'index.html'
)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    transformed_sms = transform_text(message)
    vector_input = vectorizer.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        prediction = "🚨 Spam Message Detected"
    else:
        prediction = "✅ Safe Message"

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)