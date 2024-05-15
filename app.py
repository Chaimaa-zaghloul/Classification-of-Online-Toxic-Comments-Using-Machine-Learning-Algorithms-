import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_login import LoginManager, login_required

app = Flask(__name__)


# Load the model and vectorizer
def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

def load_model(model_type):
    filename = f'{model_type}_model.pkl'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Initialize vectorizer and models
vectorizer = load_vectorizer()
models = {
    'logistic': load_model('logistic'),
    'random_forest': load_model('random_forest'),
    'naive_bayes': load_model('naive_bayes'),
    'decision_tree': load_model('decision_tree'),
    'knn': load_model('knn')
}

# Text preprocessing function
def preprocess_comment(comment):
    # Add your text preprocessing steps here
    return comment.lower()

# Function to predict with multiple models
def predict_comment(comment, models, vectorizer):
    preprocessed_comment = preprocess_comment(comment)
    comment_vector = vectorizer.transform([preprocessed_comment])
    results = {}
    for model_name, model in models.items():
        prediction = model.predict(comment_vector)[0]
        probability = model.predict_proba(comment_vector).max() if hasattr(model, "predict_proba") else "N/A"
        results[model_name] = {'prediction': prediction, 'probability': probability}
    return results

# Route to handle comment submission
@app.route('/classify', methods=['POST'])
def classify_comment():
    comment = request.form['comment']
    results = predict_comment(comment, models, vectorizer)
    return render_template('result.html', results=results)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
