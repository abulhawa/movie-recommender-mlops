from airflow.sdk import Asset

RATING_PARTS = Asset(
    name='rating_parts',
    uri='dagshub-bucket://movie-recommender-mlops/ratings_parts',
)
