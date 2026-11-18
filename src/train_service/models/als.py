"""Implicit ALS baseline on signed, rating-derived confidence values."""


def fit(data, *, factors=32, iterations=10, regularization=0.1, seed=42, threads=2):
    from implicit.cpu.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=seed,
        num_threads=threads,
    )
    model.fit(data.confidence_matrix(), show_progress=False)
    return model
