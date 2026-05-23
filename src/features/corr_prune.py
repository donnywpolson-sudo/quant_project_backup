import numpy as np

def correlation_prune(df, feature_cols, threshold=0.90):
    """
    Deterministic correlation pruning.
    Keeps first occurrence, drops later correlated features.
    """
    X = df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float64)

    corr = np.corrcoef(X, rowvar=False)

    keep = []
    dropped = set()

    for i, f in enumerate(feature_cols):
        if f in dropped:
            continue

        keep.append(f)

        for j in range(i + 1, len(feature_cols)):
            if abs(corr[i, j]) > threshold:
                dropped.add(feature_cols[j])

    return keep