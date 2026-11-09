"""Deterministic leave-one-liked-work-out evaluation against a popularity baseline."""
import argparse
import json

import numpy as np

from build_artifacts import build_pivot, cosine_similarity, load_ratings


def evaluate(merged, seed=42, k=9):
    # Split whole user/work pairs so another edition cannot leak a held-out rating.
    pairs = merged.groupby(['User-ID', 'Book-Title'], as_index=False)['Book-Rating'].max()
    liked = pairs[pairs['Book-Rating'] >= 8]
    counts = liked.groupby('User-ID').size()
    eligible = liked[liked['User-ID'].isin(counts[counts >= 5].index)]
    held = eligible.groupby('User-ID', group_keys=False).sample(n=1, random_state=seed)
    hidden = set(zip(held['User-ID'], held['Book-Title']))
    train = pairs[[pair not in hidden for pair in zip(pairs['User-ID'], pairs['Book-Title'])]]
    pivot = build_pivot(train)
    if pivot.empty or held.empty:
        raise ValueError('Not enough training ratings or eligible users to evaluate.')
    scores = cosine_similarity(pivot.values)
    positions = {title: i for i, title in enumerate(pivot.index)}
    popularity = (train[train['Book-Rating'] >= 8].groupby('Book-Title').size()
                  .reindex(pivot.index, fill_value=0).to_numpy(dtype=float))
    hits = {'collaborative': [], 'popularity': []}
    discounts = {'collaborative': [], 'popularity': []}
    covered = 0
    users = {user: rows for user, rows in train.groupby('User-ID')}
    for user, target in held[['User-ID', 'Book-Title']].itertuples(index=False, name=None):
        rows = users[user]
        seen = [positions[t] for t in rows['Book-Title'] if t in positions]
        seeds = [positions[t] for t in rows.loc[rows['Book-Rating'] >= 8, 'Book-Title'] if t in positions]
        covered += int(target in positions and bool(seeds))
        model = scores[seeds].mean(axis=0) if seeds else popularity.copy()
        for name, values in [('collaborative', model), ('popularity', popularity.copy())]:
            values = values.copy()
            values[seen] = -np.inf
            ranked = np.argsort(-values, kind='stable')[:k]
            titles = [pivot.index[i] for i in ranked if np.isfinite(values[i])]
            rank = titles.index(target) + 1 if target in titles else None
            hits[name].append(float(rank is not None))
            discounts[name].append(1 / np.log2(rank + 1) if rank else 0.)
    return {'seed': seed, 'k': k, 'held_out_users': len(held), 'training_works': len(pivot),
            'target_and_seed_coverage': covered / len(held),
            'metrics': {name: {f'recall@{k}': float(np.mean(hits[name])),
                               f'ndcg@{k}': float(np.mean(discounts[name]))} for name in hits}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='evaluation.json')
    args = parser.parse_args()
    result = evaluate(load_ratings(), seed=args.seed)
    with open(args.output, 'w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))
