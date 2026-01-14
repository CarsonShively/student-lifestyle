CREATE SCHEMA IF NOT EXISTS gold;

CREATE OR REPLACE VIEW gold.student_lifestyle AS
WITH base AS (
  SELECT
    student_id,
    study_hours_per_day             AS study,
    extracurricular_hours_per_day   AS extra,
    sleep_hours_per_day             AS sleep,
    social_hours_per_day            AS social,
    physical_activity_hours_per_day AS pa,
    gpa,
    stress_level
  FROM silver.student_lifestyle
),
fe AS (
  SELECT
    student_id, study, extra, sleep, social, pa, gpa, stress_level,

    total_active(study, extra, social, pa)            AS total_active,
    time_balance(sleep, study, extra, social, pa)     AS time_balance,
    overscheduled(sleep, study, extra, social, pa)    AS overscheduled,

    study_share(study, sleep)                         AS study_share,
    social_share(social, sleep)                       AS social_share,
    extra_share(extra, sleep)                         AS extra_share,
    pa_share(pa, sleep)                               AS pa_share,

    study_to_sleep(study, sleep)                      AS study_to_sleep,
    study_to_social(study, social)                    AS study_to_social,

    sleep_deficit(sleep)                              AS sleep_deficit,
    sleep_abs_delta(sleep)                            AS sleep_abs_delta,
    sleep_low(sleep)                                  AS sleep_low,
    sleep_high(sleep)                                 AS sleep_high,

    pa_guideline(pa)                                  AS pa_guideline,
    pa_under(pa)                                      AS pa_under,

    study_c2(study)                                   AS study_c2,
    sleep_c2(sleep)                                   AS sleep_c2,
    social_c2(social)                                 AS social_c2,

    sleep_x_study(sleep, study)                       AS sleep_x_study,
    study_x_social(study, social)                     AS study_x_social
  FROM base
)
SELECT * FROM fe;
