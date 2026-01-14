CREATE OR REPLACE MACRO flag(expr) AS CASE WHEN (expr) THEN 1 ELSE 0 END;
CREATE OR REPLACE MACRO safe_div(num, den) AS (num / NULLIF(den, 0));
CREATE OR REPLACE MACRO awake_hours(sleep) AS (24 - sleep);

CREATE OR REPLACE MACRO total_active(study, extra, social, pa)
AS (study + extra + social + pa);

CREATE OR REPLACE MACRO time_balance(sleep, study, extra, social, pa)
AS (24 - (sleep + study + extra + social + pa));

CREATE OR REPLACE MACRO overscheduled(sleep, study, extra, social, pa)
AS flag((sleep + study + extra + social + pa) > 24);

CREATE OR REPLACE MACRO study_share(study, sleep)
AS safe_div(study, awake_hours(sleep));

CREATE OR REPLACE MACRO social_share(social, sleep)
AS safe_div(social, awake_hours(sleep));

CREATE OR REPLACE MACRO extra_share(extra, sleep)
AS safe_div(extra,  awake_hours(sleep));

CREATE OR REPLACE MACRO pa_share(pa, sleep)
AS safe_div(pa,     awake_hours(sleep));

CREATE OR REPLACE MACRO study_to_sleep(study, sleep)
AS safe_div(study, sleep);

CREATE OR REPLACE MACRO study_to_social(study, social)
AS safe_div(study, social);

CREATE OR REPLACE MACRO sleep_deficit(sleep)
AS GREATEST(0, 8 - sleep);

CREATE OR REPLACE MACRO sleep_abs_delta(sleep)
AS ABS(sleep - 8);

CREATE OR REPLACE MACRO sleep_low(sleep)
AS flag(sleep < 7);

CREATE OR REPLACE MACRO sleep_high(sleep)
AS flag(sleep > 9);

CREATE OR REPLACE MACRO pa_guideline(pa)
AS flag(pa >= 1.0);

CREATE OR REPLACE MACRO pa_under(pa)
AS GREATEST(0, 1.0 - pa);

CREATE OR REPLACE MACRO study_c2(study)
AS POW(study - 4, 2);

CREATE OR REPLACE MACRO sleep_c2(sleep)
AS POW(sleep - 8, 2);

CREATE OR REPLACE MACRO social_c2(social)
AS POW(social - 2, 2);

CREATE OR REPLACE MACRO sleep_x_study(sleep, study)
AS (sleep * study);

CREATE OR REPLACE MACRO study_x_social(study, social)
AS (study * social);