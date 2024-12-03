from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from flask_cors import CORS
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)


df = pd.read_csv("recipe.csv")

vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(df['ingredients_list'])


knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_ingredients)

def recommend_recipes(input_features):
    input_ingredients_transformed = vectorizer.transform([input_features[0]])
    distances, indices = knn.kneighbors(X_ingredients)
    recommendations = df.iloc[indices[0]]
    return recommendations[['recipe_name', 'ingredients_list', 'image_url']]

def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ingredients = request.form['ingredients']
        input_features = [ingredients]
        recommendations = recommend_recipes(input_features)
        return render_template('index.html', recommendations=recommendations.to_dict(orient='records'),truncate = truncate)
    return render_template('index.html', recommendations=[])

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
