import pandas as pd

def update_wins(mtx: dict[int,dict[int,int]], loser: int, cur_winner: int):
    for candidate in mtx[cur_winner]:
        if mtx[cur_winner][candidate] == 1:
            mtx[candidate][loser] = 1
            mtx[loser][candidate] = -1
            update_wins(mtx, loser, candidate)

def update_losses(mtx: dict[int,dict[int,int]], winner: int, cur_loser: int):
    for candidate in mtx[cur_loser]:
        if mtx[cur_loser][candidate] == -1:
            mtx[candidate][winner] = -1
            mtx[winner][candidate] = 1
            update_losses(mtx, winner, candidate)

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
    while len(order) < len(idxs):
        for idx, scores in mtx.items():
            current_visit = set()
            if all((x == 1 for i,x in scores.items() if i not in visited)):
                order.append(idx)
                current_visit.add(idx)
            visited.update(current_visit)
    return order
