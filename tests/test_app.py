import os
import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as book_app  # noqa: E402
from titles import clean_text, normalize_title  # noqa: E402


@pytest.fixture
def client():
    return book_app.app.test_client()


def search(client, title):
    return client.post('/recommend_books', data={'user_input': title})


def test_artifact_filenames_have_exact_case():
    # Linux is case-sensitive, even when the development machine isn't
    names = os.listdir(ROOT)
    for name in ['books.pkl', 'popular_df.pkl', 'pt.pkl', 'similarity_score.pkl']:
        assert name in names


def test_home_page_lists_top_50(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.text.count('class="grid-item"') == 50


def test_top_50_uses_explicit_ratings():
    assert book_app.popular_df['avg_ratings'].min() > 7
    assert book_app.popular_df['weighted_rating'].is_monotonic_decreasing


def test_results_are_repeatable(client):
    first = search(client, '1984')
    second = search(client, '1984')
    assert first.status_code == 200
    assert first.text.count('class="grid-item"') == 9
    assert first.text == second.text


@pytest.mark.parametrize('variant', ['the hours: a novel', '  The Hours : A Novel ', 'THE HOURS A NOVEL'])
def test_search_ignores_case_spacing_and_punctuation(client, variant):
    assert search(client, variant).text.count('class="grid-item"') == 9


def test_book_is_not_recommended_to_itself(client):
    title = book_app.pt.index[0]
    html = search(client, title).text
    results = html.split('class="grid-container"')[1]
    assert f'<p class="book-title">{title}</p>' not in results


def test_unknown_book_shows_error(client):
    response = search(client, 'zzz not a book')
    assert response.status_code == 404
    assert 'Book not found' in response.text


def test_blank_search_shows_error(client):
    response = search(client, '   ')
    assert response.status_code == 400
    assert 'Please enter a book title' in response.text


def test_autocomplete_without_query_returns_empty_list(client):
    response = client.get('/autocomplete')
    assert response.status_code == 200
    assert response.json == []


def test_autocomplete_finds_books_outside_the_top_50(client):
    top_50 = set(book_app.popular_df['Book-Title'])
    title = next(title for title in book_app.pt.index if title not in top_50)
    assert title in client.get('/autocomplete', query_string={'q': title.lower()}).json


def test_autocomplete_suggestions_all_return_results(client):
    for title in client.get('/autocomplete', query_string={'q': 'harry'}).json:
        assert search(client, title).status_code == 200


def test_every_recommendable_book_has_details():
    assert set(book_app.pt.index) <= set(book_app.books.index)
    assert not book_app.books.isna().any().any()


def test_data_has_no_duplicates_or_html_entities():
    keys = [normalize_title(title) for title in book_app.pt.index]
    assert len(keys) == len(set(keys))
    assert all(clean_text(title) == title for title in book_app.pt.index)


def test_covers_use_https():
    assert book_app.books['Image-URL-M'].str.startswith('https://').all()
    assert book_app.popular_df['Image-URL-M'].str.startswith('https://').all()


@pytest.mark.parametrize('query', ['THE HOURS A NOVEL', ' The Hours : A Novel ', 'the hours: a novel'])
def test_autocomplete_normalizes_like_search(client, query):
    suggestions = client.get('/autocomplete', query_string={'q': query}).json
    assert 'The Hours: A Novel' in suggestions


def test_punctuation_only_autocomplete_is_empty(client):
    assert client.get('/autocomplete', query_string={'q': '?!'}).json == []


def test_long_title_is_rejected(client):
    assert search(client, 'a' * 501).status_code == 400
    assert client.get('/autocomplete', query_string={'q': 'a' * 501}).status_code == 400


def test_large_request_has_friendly_error(client):
    response = search(client, 'a' * (17 * 1024))
    assert response.status_code == 413
    assert 'Search request is too large' in response.text


@pytest.mark.parametrize('scores', [np.eye(1), np.array([[1, np.nan], [np.nan, 1]]),
                                  np.array([[1, 0.5], [0, 1]])])
def test_invalid_similarity_artifacts_are_rejected(scores):
    pivot = pd.DataFrame(index=['First', 'Second'])
    catalog = pd.DataFrame({'Book-Author': ['A', 'B'], 'Image-URL-M': ['a', 'b']},
                           index=pivot.index)
    with pytest.raises(ValueError, match='Similarity matrix'):
        book_app.validate_artifacts(pivot, catalog, scores)


def test_missing_catalog_entry_is_rejected():
    with pytest.raises(ValueError, match='missing recommendation titles'):
        book_app.validate_artifacts(book_app.pt, book_app.books.iloc[:0], book_app.similarity_score)


def test_every_homepage_title_has_a_working_discovery_link(client):
    for title in book_app.popular_df['Book-Title']:
        response = client.get('/recommend', query_string={'title': title})
        assert response.status_code == 200
        assert response.text.count('class="grid-item"') == 9


def test_unsupported_popular_book_has_an_honest_fallback(client):
    title = next(t for t in book_app.popular_df['Book-Title'] if t not in book_app.pt.index)
    response = search(client, title)
    assert response.status_code == 200
    assert 'enough shared reader ratings' in response.text
    assert 'Because you liked' not in response.text


def test_detailed_autocomplete_includes_author_and_searchable_title(client):
    suggestions = client.get('/autocomplete', query_string={'q': 'harry', 'details': '1'}).json
    assert suggestions
    for suggestion in suggestions:
        assert suggestion['author']
        assert search(client, suggestion['title']).status_code == 200


def test_get_search_uses_the_same_input_limits(client):
    assert client.get('/recommend', query_string={'title': 'a' * 501}).status_code == 400
    assert client.get('/recommend', query_string={'title': '<script>alert(1)</script>'}).status_code == 404
    assert '<script>alert(1)</script>' not in search(client, '<script>alert(1)</script>').text
