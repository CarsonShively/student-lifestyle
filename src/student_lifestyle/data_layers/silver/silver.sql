CREATE SCHEMA IF NOT EXISTS silver;

CREATE OR REPLACE VIEW silver.student_lifestyle AS
WITH typed AS (
  SELECT
    CAST(Student_ID AS BIGINT)                      AS student_id,
    CAST(Study_Hours_Per_Day AS DOUBLE)             AS study_hours_per_day,
    CAST(Extracurricular_Hours_Per_Day AS DOUBLE)   AS extracurricular_hours_per_day,
    CAST(Sleep_Hours_Per_Day AS DOUBLE)             AS sleep_hours_per_day,
    CAST(Social_Hours_Per_Day AS DOUBLE)            AS social_hours_per_day,
    CAST(Physical_Activity_Hours_Per_Day AS DOUBLE) AS physical_activity_hours_per_day,
    CAST(GPA AS DOUBLE)                             AS gpa,
    CASE
      WHEN lower(trim(Stress_Level)) = 'low'      THEN 'Low'
      WHEN lower(trim(Stress_Level)) = 'moderate' THEN 'Moderate'
      WHEN lower(trim(Stress_Level)) = 'high'     THEN 'High'
      ELSE NULL
    END::VARCHAR                                    AS stress_level
  FROM bronze.student_lifestyle
),
validated AS (
  SELECT *
  FROM typed
  WHERE
    study_hours_per_day                 IS NOT NULL AND study_hours_per_day                 BETWEEN 0 AND 24 AND
    extracurricular_hours_per_day       IS NOT NULL AND extracurricular_hours_per_day       BETWEEN 0 AND 24 AND
    sleep_hours_per_day                 IS NOT NULL AND sleep_hours_per_day                 BETWEEN 0 AND 24 AND
    social_hours_per_day                IS NOT NULL AND social_hours_per_day                BETWEEN 0 AND 24 AND
    physical_activity_hours_per_day     IS NOT NULL AND physical_activity_hours_per_day     BETWEEN 0 AND 24 AND
    gpa                                 IS NOT NULL AND gpa                                 BETWEEN 0 AND 4  AND
    stress_level                        IS NOT NULL
),
deduped AS (
  SELECT DISTINCT
    student_id,
    study_hours_per_day,
    extracurricular_hours_per_day,
    sleep_hours_per_day,
    social_hours_per_day,
    physical_activity_hours_per_day,
    gpa,
    stress_level
  FROM validated
)
SELECT *
FROM deduped;
