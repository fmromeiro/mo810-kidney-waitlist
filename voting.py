import pandas as pd

def update_wins(mtx, loser, cur_winner):
    for candidate in mtx[cur_winner]:
        if mtx[cur_winner][candidate] == 1:
            mtx[candidate][loser] = 1
            mtx[loser][candidate] = -1
            update_wins(loser, candidate)

def update_losses(mtx, winner, cur_loser):
    for candidate in mtx[cur_loser]:
        if mtx[cur_loser][candidate] == -1:
            mtx[candidate][winner] = -1
            mtx[winner][candidate] = 1
            update_wins(winner, candidate)

def voting_score(ps: pd.Series, meta: pd.DataFrame, threshold: float = 0.5):
    df = meta.copy()
    df['p'] = ps
    df.sort_values('p',inplace=True)

    idxs = set()
    idxs.update(df['idx_a'].to_list())
    idxs.update(df['idx_b'].to_list())

    mtx = {i:{} for i in idxs}

    for _, row in df.iterrows():
        idx_a = row['idx_a']
        idx_b = row['idx_b']
        if idx_b in mtx[idx_a]:
            continue

        p = row['p']
        winner, loser = 0
        if p < threshold:
            winner, loser = idx_a, idx_b
        else:
            winner, loser = idx_b, idx_a

        mtx[winner][loser] = 1
        mtx[loser][winner] = -1
        update_wins(mtx,loser,winner)
        update_losses(mtx,winner,loser)

    order = []
    visited = set()
    for idx, scores in mtx.items():
        if all((x == 1 for i,x in scores.items() if i not in visited)):
            order.append(idx)
            visited.add(idx)
    return order
