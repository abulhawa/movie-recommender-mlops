from airflow.sdk import Asset

# Event identity shared by the uploader and training DAG; this URI labels the
# event. Actual bucket access uses the scheduler's DAGSHUB_* environment values.
RATING_PARTS = Asset(
    name='rating_parts',
    uri='dagshub-bucket://movie-recommender-mlops/ratings_parts',
)
