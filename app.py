from pathlib import Path

from flask import Flask, render_template, request, jsonify
import numpy as np

from titles import normalize_title
from artifacts import load_bundle

BASE_DIR = Path(__file__).resolve().parent
NUM_RECOMMENDATIONS = 9
NUM_SUGGESTIONS = 10
MAX_TITLE_LENGTH = 500


snapshot, artifact_manifest = load_bundle(BASE_DIR / 'recommendation_bundle.pkl')
popular_df = snapshot['popular']
pt = snapshot['pivot']
books = snapshot['catalog'].set_index('Book-Title')
similarity_score = snapshot['scores']


def validate_artifacts(pivot, catalog, scores):
    count = len(pivot.index)
    if not count or not pivot.index.is_unique or not catalog.index.is_unique:
        raise ValueError('Artifacts require nonempty, unique book titles.')
    if not set(pivot.index) <= set(catalog.index):
        raise ValueError('Catalog is missing recommendation titles.')
    if not {'Book-Author', 'Image-URL-M'} <= set(catalog.columns):
        raise ValueError('Catalog is missing author or cover columns.')
    if scores.shape != (count, count) or not np.isfinite(scores).all():
        raise ValueError('Similarity matrix has invalid dimensions or values.')
    if not np.allclose(scores, scores.T):
        raise ValueError('Similarity matrix must be symmetric.')
    keys = [normalize_title(title) for title in pivot.index]
    if len(set(keys)) != count or not all(keys):
        raise ValueError('Recommendation titles have empty or duplicate search keys.')


validate_artifacts(pt, books, similarity_score)
title_positions = {normalize_title(title): i for i, title in enumerate(pt.index)}
search_titles = [(title, normalize_title(title)) for title in books.index]
catalog_titles = {normalize_title(title): title for title in books.index}

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=16 * 1024, MAX_FORM_MEMORY_SIZE=16 * 1024,
                  MAX_FORM_PARTS=10)


@app.errorhandler(413)
def request_too_large(error):
    return render_template('recommend.html', error='Search request is too large.'), 413


@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=popular_df['avg_ratings'].to_list()
                           )


@app.route('/recommend')
def recommend_ui():
    if request.args.get('title'):
        return render_recommendations(request.args['title'].strip())
    return render_template('recommend.html')


@app.route("/recommend_books", methods=['POST'])
def recommend():
    return render_recommendations(request.form.get('user_input', '').strip())


def render_recommendations(user_input):
    if not user_input:
        return render_template('recommend.html', error="Please enter a book title."), 400

    if len(user_input) > MAX_TITLE_LENGTH:
        return render_template('recommend.html', error='Book titles must be 500 characters or fewer.'), 400

    position = title_positions.get(normalize_title(user_input))
    if position is None and normalize_title(user_input) in catalog_titles:
        title = catalog_titles[normalize_title(user_input)]
        fallback = popular_df[popular_df['Book-Title'] != title].head(NUM_RECOMMENDATIONS)
        data = fallback[['Book-Title', 'Book-Author', 'Image-URL-M']].values.tolist()
        return render_template('recommend.html', data=data, query=title, fallback=True)
    if position is None:
        return render_template('recommend.html', query=user_input,
                               error="Book not found. Please pick a title from the suggestions."), 404

    ranked = np.argsort(-similarity_score[position], kind='stable')
    similar_items = [i for i in ranked if i != position][:NUM_RECOMMENDATIONS]

    data = []
    for i in similar_items:
        title = pt.index[i]
        data.append([title, books.at[title, 'Book-Author'], books.at[title, 'Image-URL-M']])

    return render_template('recommend.html', data=data, query=pt.index[position])


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('q', '')
    if len(query) > MAX_TITLE_LENGTH:
        return jsonify([]), 400
    search = normalize_title(query)
    if not search:
        return jsonify([])
    matches = [(title, key) for title, key in search_titles if search in key]
    # Titles starting with the search text come first
    matches.sort(key=lambda item: not item[1].startswith(search))
    titles = [title for title, _ in matches[:NUM_SUGGESTIONS]]
    if request.args.get('details') == '1':
        return jsonify([{'title': title, 'author': books.at[title, 'Book-Author']} for title in titles])
    return jsonify(titles)


if __name__ == '__main__':
    app.run(port=5000)
