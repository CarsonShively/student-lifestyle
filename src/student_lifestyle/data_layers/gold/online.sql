CREATE OR REPLACE MACRO gold_online_row(
  study, extra, sleep, social, pa, gpa
) AS TABLE
SELECT
  CAST(study  AS DOUBLE) AS study,
  CAST(extra  AS DOUBLE) AS extra,
  CAST(sleep  AS DOUBLE) AS sleep,
  CAST(social AS DOUBLE) AS social,
  CAST(pa     AS DOUBLE) AS pa,
  CAST(gpa    AS DOUBLE) AS gpa,

  total_active(study, extra, social, pa)         AS total_active,
  time_balance(sleep, study, extra, social, pa)  AS time_balance,
  overscheduled(sleep, study, extra, social, pa) AS overscheduled,

  study_share(study, sleep)                      AS study_share,
  social_share(social, sleep)                    AS social_share,
  extra_share(extra, sleep)                      AS extra_share,
  pa_share(pa, sleep)                            AS pa_share,

  study_to_sleep(study, sleep)                   AS study_to_sleep,
  study_to_social(study, social)                 AS study_to_social,

  sleep_deficit(sleep)                           AS sleep_deficit,
  sleep_abs_delta(sleep)                         AS sleep_abs_delta,
  sleep_low(sleep)                               AS sleep_low,
  sleep_high(sleep)                              AS sleep_high,

  pa_guideline(pa)                               AS pa_guideline,
  pa_under(pa)                                   AS pa_under,

  study_c2(study)                                AS study_c2,
  sleep_c2(sleep)                                AS sleep_c2,
  social_c2(social)                              AS social_c2,

  sleep_x_study(sleep, study)                    AS sleep_x_study,
  study_x_social(study, social)                  AS study_x_social;