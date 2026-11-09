import pandas as pd
import build_artifacts
from evaluate import evaluate


def test_evaluation_is_repeatable_and_masks_seen_books(monkeypatch):
    monkeypatch.setattr(build_artifacts, 'MIN_USER_RATINGS', 0)
    monkeypatch.setattr(build_artifacts, 'MIN_BOOK_RATINGS', 1)
    # Every user knows all five works. After withholding one, it is the only unseen candidate.
    rows = pd.DataFrame([(user, title, 9) for user in ['a', 'b', 'c']
                         for title in ['One', 'Two', 'Three', 'Four', 'Five']],
                        columns=['User-ID', 'Book-Title', 'Book-Rating'])
    first = evaluate(rows, k=1)
    assert first == evaluate(rows, k=1)
    assert first['held_out_users'] == 3
    assert first['target_and_seed_coverage'] == 1
    assert first['metrics']['collaborative']['recall@1'] == 1
    assert first['metrics']['popularity']['ndcg@1'] == 1


def test_evaluation_rejects_insufficient_data():
    import pytest
    rows = pd.DataFrame([('a', 'One', 9)], columns=['User-ID', 'Book-Title', 'Book-Rating'])
    with pytest.raises(ValueError, match='Not enough'):
        evaluate(rows)
