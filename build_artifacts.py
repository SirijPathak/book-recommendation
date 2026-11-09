"""Rebuild the atomic artifact snapshot used by app.py."""
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from titles import clean_text, normalize_title
from artifacts import save_bundle

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / 'data' / 'source'

MIN_USER_RATINGS = 200
MIN_BOOK_RATINGS = 50
TOP_N = 50
TOP_MIN_VOTES = 50
TOP_PRIOR_VOTES = 50
COVER_URL = 'https://covers.openlibrary.org/b/isbn/{}-M.jpg?default=false'


def prepare_ratings(books, ratings):
    """Validate records and merge editions only when title AND author agree."""
    books = books[['ISBN', 'Book-Title', 'Book-Author']].copy()
    books = books.dropna(subset=['ISBN', 'Book-Title'])
    books['Book-Title'] = books['Book-Title'].map(clean_text)
    books['Book-Author'] = books['Book-Author'].fillna('Unknown author').map(clean_text)
    books.loc[books['Book-Author'] == '', 'Book-Author'] = 'Unknown author'
    books['key'] = books['Book-Title'].map(normalize_title)
    books['author_key'] = books['Book-Author'].map(normalize_title)
    books = books[books['key'] != ''].drop_duplicates('ISBN')
    ratings = ratings.dropna(subset=['User-ID', 'ISBN']).copy()
    ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce')
    ratings = ratings[ratings['Book-Rating'].between(0, 10)]
    ratings = ratings.groupby(['User-ID', 'ISBN'], as_index=False)['Book-Rating'].max()
    merged = ratings.merge(books, on='ISBN', validate='many_to_one')
    merged['Work-ID'] = merged['key'] + ':' + merged['author_key']
    titles = (merged.groupby(['Work-ID', 'Book-Title', 'Book-Author']).size().rename('n').reset_index()
              .sort_values(['n', 'Book-Title', 'Book-Author'], ascending=[False, True, True])
              .drop_duplicates('Work-ID'))
    collisions = titles['Book-Title'].map(normalize_title).duplicated(keep=False)
    titles.loc[collisions, 'Book-Title'] = (titles.loc[collisions, 'Book-Title'] + ' by '
                                         + titles.loc[collisions, 'Book-Author'])
    # Explicit labels remain unique even for unusual punctuation-only author names.
    ambiguous = titles['Book-Title'].map(normalize_title).duplicated(keep=False)
    titles.loc[ambiguous, 'Book-Title'] += titles.loc[ambiguous, 'Work-ID'].map(
        lambda work: ' [' + hashlib.sha256(work.encode()).hexdigest()[:12] + ']')
    merged['Book-Title'] = merged['Work-ID'].map(titles.set_index('Work-ID')['Book-Title'])
    return merged


def load_ratings():
    with open(SOURCE_DIR / 'books.pkl', 'rb') as f:
        books = pickle.load(f)
    ratings = pd.read_csv(SOURCE_DIR / 'Ratings.csv', sep=';', encoding='latin-1',
                          dtype={'User-ID': str, 'ISBN': str})
    return prepare_ratings(books, ratings)


def build_catalog(merged):
    # Show the author and cover of each book's most-rated edition
    editions = (merged.groupby(['Book-Title', 'ISBN', 'Book-Author']).size().rename('n').reset_index()
                .sort_values(['n', 'ISBN'], ascending=[False, True]).drop_duplicates('Book-Title'))
    editions['Image-URL-M'] = editions['ISBN'].map(COVER_URL.format)
    return editions[['Book-Title', 'Book-Author', 'Image-URL-M']].set_index('Book-Title')


def build_popular(merged, catalog):
    # A rating of 0 in this dataset means "no explicit rating", so it must not lower the average
    explicit = merged[merged['Book-Rating'] > 0]
    if 'User-ID' in explicit.columns:
        explicit = explicit.groupby(['User-ID', 'Book-Title'], as_index=False)['Book-Rating'].max()
    stats = explicit.groupby('Book-Title')['Book-Rating'].agg(num_ratings='count', avg_ratings='mean')
    stats = stats[stats['num_ratings'] >= TOP_MIN_VOTES]
    prior = explicit['Book-Rating'].mean()
    # Pull averages towards the overall mean so books with few votes can't top the list
    stats['weighted_rating'] = ((stats['num_ratings'] * stats['avg_ratings'] + TOP_PRIOR_VOTES * prior)
                                / (stats['num_ratings'] + TOP_PRIOR_VOTES))
    top = stats.sort_values('weighted_rating', ascending=False).head(TOP_N)
    return top.join(catalog).reset_index()


def build_pivot(merged):
    merged = merged.groupby(['User-ID', 'Book-Title'], as_index=False)['Book-Rating'].max()
    user_counts = merged.groupby('User-ID').size()
    active = merged[merged['User-ID'].isin(user_counts[user_counts > MIN_USER_RATINGS].index)]
    book_counts = active.groupby('Book-Title').size()
    active = active[active['Book-Title'].isin(book_counts[book_counts >= MIN_BOOK_RATINGS].index)]
    return active.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating',
                              aggfunc='max', fill_value=0).astype(float)


def cosine_similarity(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit = matrix / norms
    return unit @ unit.T


def main():
    merged = load_ratings()
    catalog = build_catalog(merged)
    popular_df = build_popular(merged, catalog)
    pt = build_pivot(merged)
    similarity_score = cosine_similarity(pt.values)
    books = catalog[catalog.index.isin(set(pt.index) | set(popular_df['Book-Title']))].reset_index()

    save_bundle(BASE_DIR / 'recommendation_bundle.pkl', popular_df, pt, books, similarity_score)
    print('Published recommendation_bundle.pkl:', len(pt), 'recommendable works,', len(books), 'searchable works')


if __name__ == '__main__':
    main()
