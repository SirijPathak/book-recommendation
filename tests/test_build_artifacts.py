import numpy as np
import pandas as pd

from build_artifacts import build_popular, cosine_similarity, prepare_ratings, build_pivot
import build_artifacts
import pytest
from artifacts import save_bundle, load_bundle


def test_cosine_similarity_handles_known_vectors_and_unrated_books():
    scores = cosine_similarity(np.array([[3., 4.], [6., 8.], [4., -3.], [0., 0.]]))
    np.testing.assert_allclose(scores, [[1, 1, 0, 0], [1, 1, 0, 0],
                                      [0, 0, 1, 0], [0, 0, 0, 0]], atol=1e-12)


def test_popularity_ignores_implicit_ratings_and_shrinks_small_samples():
    merged = pd.DataFrame({'Book-Title': ['Small'] * 50 + ['Large'] * 100 + ['Implicit'] * 100,
                           'Book-Rating': [8] * 50 + [8] * 100 + [0] * 100})
    # Add explicit ratings below the eligibility threshold to establish a lower prior.
    merged = pd.concat([merged, pd.DataFrame({'Book-Title': ['Other'] * 10,
                                             'Book-Rating': [1] * 10})])
    catalog = pd.DataFrame({'Book-Author': ['A', 'B', 'C', 'D'],
                            'Image-URL-M': ['a', 'b', 'c', 'd']},
                           index=['Small', 'Large', 'Implicit', 'Other'])
    catalog.index.name = 'Book-Title'
    popular = build_popular(merged, catalog)
    assert popular['Book-Title'].tolist() == ['Large', 'Small']
    assert popular['num_ratings'].tolist() == [100, 50]
    assert popular['avg_ratings'].tolist() == [8, 8]
    prior = (150 * 8 + 10) / 160
    np.testing.assert_allclose(popular['weighted_rating'],
                               [(100 * 8 + 50 * prior) / 150, (50 * 8 + 50 * prior) / 100])


def test_editions_merge_but_different_authors_stay_separate():
    books = pd.DataFrame({'ISBN': ['1', '2', '3'], 'Book-Title': ['Same!', 'same', 'Same!'],
                          'Book-Author': ['Alice', 'ALICE', 'Bob']})
    ratings = pd.DataFrame({'User-ID': ['u', 'u', 'u'], 'ISBN': ['1', '2', '3'],
                            'Book-Rating': [7, 8, 9]})
    result = prepare_ratings(books, ratings)
    assert result['Work-ID'].nunique() == 2
    assert result.iloc[0]['Book-Title'] == result.iloc[1]['Book-Title']
    assert result.iloc[0]['Book-Title'] != result.iloc[2]['Book-Title']


def test_invalid_records_and_duplicate_user_ratings():
    books = pd.DataFrame({'ISBN': ['1', '2'], 'Book-Title': ['Valid', None],
                          'Book-Author': [None, 'A']})
    ratings = pd.DataFrame({'User-ID': ['u', 'u', 'x', 'y', 'z', None, 'v'],
                            'ISBN': ['1', '1', '1', '1', '2', '1', 'missing'],
                            'Book-Rating': [3, 9, -1, 'bad', 8, 4, 5]})
    result = prepare_ratings(books, ratings)
    assert result['Book-Rating'].tolist() == [9]
    assert result['Book-Author'].tolist() == ['Unknown author']


def test_filter_thresholds_and_edition_ratings(monkeypatch):
    monkeypatch.setattr(build_artifacts, 'MIN_USER_RATINGS', 2)
    monkeypatch.setattr(build_artifacts, 'MIN_BOOK_RATINGS', 2)
    rows = pd.DataFrame({'User-ID': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c'],
                         'Book-Title': ['One', 'One', 'Two', 'Extra', 'One', 'Two', 'Rare', 'One', 'Two'],
                         'Book-Rating': [4, 9, 7, 8, 8, 6, 10, 8, 8]})
    pivot = build_pivot(rows)
    assert list(pivot.index) == ['One', 'Two']
    assert list(pivot.columns) == ['a', 'b']
    assert pivot.at['One', 'a'] == 9


def test_multiple_editions_do_not_inflate_popularity_votes():
    rows = pd.DataFrame({'User-ID': [str(i) for i in range(30)] * 2,
                         'Book-Title': ['One'] * 60, 'Book-Rating': [8] * 30 + [9] * 30})
    catalog = pd.DataFrame({'Book-Author': ['A'], 'Image-URL-M': ['cover']}, index=['One'])
    catalog.index.name = 'Book-Title'
    assert build_popular(rows, catalog).empty


def test_snapshot_roundtrip_and_title_order_protection(tmp_path):
    path = tmp_path / 'bundle.pkl'
    pivot = pd.DataFrame([[1, 2], [3, 4]], index=['One', 'Two'])
    save_bundle(path, pd.DataFrame(), pivot, pd.DataFrame(), np.eye(2))
    data, manifest = load_bundle(path)
    assert manifest['build_version']
    assert list(data['pivot'].index) == ['One', 'Two']
    import pickle
    import hashlib
    bundle = pickle.loads(path.read_bytes())
    data['score_titles'].reverse()
    bundle['payload'] = pickle.dumps(data)
    bundle['manifest']['payload_sha256'] = hashlib.sha256(bundle['payload']).hexdigest()
    path.write_bytes(pickle.dumps(bundle))
    with pytest.raises(ValueError, match='title order'):
        load_bundle(path)


def test_snapshot_detects_corruption_and_preserves_previous_build(tmp_path, monkeypatch):
    import artifacts
    import pickle
    path = tmp_path / 'bundle.pkl'
    pivot = pd.DataFrame(index=['One'])
    save_bundle(path, pivot, pivot, pivot, np.eye(1))
    previous = path.read_bytes()
    def fail_replace(*args):
        raise OSError('Simulated interrupted publication')
    monkeypatch.setattr(artifacts.os, 'replace', fail_replace)
    with pytest.raises(OSError):
        save_bundle(path, pivot, pivot, pivot, np.eye(1))
    assert path.read_bytes() == previous
    assert len(list(tmp_path.iterdir())) == 1
    bundle = pickle.loads(previous)
    bundle['payload'] += b'corruption'
    path.write_bytes(pickle.dumps(bundle))
    with pytest.raises(ValueError, match='checksum'):
        load_bundle(path)
