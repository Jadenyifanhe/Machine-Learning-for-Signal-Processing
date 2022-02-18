

def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    score = (y_pred == y_true).astype(float).sum() / len(y_true)
    return score


