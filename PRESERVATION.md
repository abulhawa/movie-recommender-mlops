# Independent preservation

Created 17 November 2026 for the owner `abulhawa`.

## Origin and authorship

- Original team GitHub: https://github.com/rosneru/sep26_bmlops_movie_reco
- Original team DagsHub: https://dagshub.com/rosneru/sep26_bmlops_movie_reco
- Original template: https://github.com/DataScientest-Studio/Template_MLOps_movie_recommandation
- Original MIT license retained unchanged. Original Git commit objects, author/committer metadata and timestamps are preserved.
- Snapshot: 176 reachable commits and eight branches; no tags existed. Original main: `58d18ba6cf7563d0d9ee8fd0413226c37d7a79fc`.
- GitHub and DagsHub source branch heads matched at preservation time.

## Owned destinations

- GitHub: https://github.com/abulhawa/sep26_bmlops_movie_reco
- DagsHub: https://dagshub.com/abulhawa/sep26_bmlops_movie_reco
- Both private, independent repositories; owner has administrator access.
- GitHub Actions disabled during preservation; no services, training or deployment started.
- Runtime defaults in this copy use the owner's DagsHub endpoints. Historical branches retain historical settings and must be adapted before execution.
- The original repositories are blocked for writes. Do not push, log experiments, register models, upload data or change their settings.

## Data preservation

Both original stores are preserved: five DVC objects (74,357,201 bytes) and nine repository-bucket objects (527,342,083 bytes). Nine additional current-pipeline DVC cache objects (986,387,678 bytes) are recovered from the original local checkout without changing it.

Four historical outputs referenced by commit `0bc1217d` are unavailable in the original remote and local cache: two directory manifests (`410269c6cb6acc61f85277a6fe85ccc6.dir`, `f9bc5cc9cb464033b5976b7910037b09.dir`) and the older processed matrices (`1554692a2965d448a2f3ae4e523abd12`, `390fcb7ce6edb5721dc8729d6d7ea613`). Their Git references are preserved; a fully reproducible data snapshot for those older revisions cannot be claimed.

## MLflow preservation

Source snapshot: one experiment, 73 runs, one registered model with 66 available versions (numbered 1–67, with version 18 absent) and four aliases. Raw source metadata is retained under `preservation/`. Migration progress and run/version mappings are recorded separately; consult the final verification below. MLflow assigns new run IDs and creation timestamps to imported records. Original IDs/timestamps are preserved as metadata; Git authorship is unaffected. Artifacts are copied without changing their bytes. Original `mlflow.user` tags have been restored after DagsHub initially replaced them with the importing account; imported records are owned by the destination account.

## Local copies

- Working copy: `/home/ali/liora/sep26_bmlops_movie_reco_personal`.
- Snapshot, Git bundle, migration checkpoints and downloaded data/artifacts: `/home/ali/liora/personal-movie-reco-backup`.
- Credentials remain in ignored local files, never in the archive metadata.

## Verification

All 73 runs, their parameters/tags/status/start times and originally present end times, 66 model versions, and four aliases were verified in the destination. Sample model artifacts passed SHA-256 download comparison. All 23 copied storage objects passed full download-and-MD5 comparison. The original working tree remains clean. See `preservation/final-verification.json` and `preservation/mlflow-mapping.json`.

DagsHub initially selected `Dev_Mani` as its default branch. Use the `main` branch for the adapted personal configuration. Changing the landing branch requires the owner settings UI; the public API did not support that setting. GitHub defaults to `main`.

Seven source runs were still marked RUNNING with no end time. Their original status is retained and raw source metadata records the missing end times; the import API assigned a placeholder end time equal to the start time. These timestamps are not evidence that the original runs ended then.

This preserves Git history and MLflow/data records; GitHub issues, pull-request discussions, workflow histories and external account settings are not part of the Git copy. Services were not started; this is preservation verification, not an end-to-end runtime test.

The original team main commit is also pinned by the tag `preservation/team-main-2026-11-17`. New personal configuration commits are appended to the preserved history.
