INSTALL httpfs;
LOAD httpfs;

CREATE SCHEMA IF NOT EXISTS bronze;

CREATE OR REPLACE VIEW bronze.student_lifestyle AS
SELECT *
FROM read_parquet(
  'https://huggingface.co/datasets/Carson-Shively/student-lifestyle/resolve/main/data/bronze/bronze.parquet'
);