-- Leakage-aware pneumonia cohort (Section 3.1, Table 2, Fig. 2).
-- Dialect: DuckDB reading the PhysioNet CSV releases directly. $HOSP, $ED and $CXR are filled in by
-- scripts/build_cohort.py with the paths of MIMIC-IV/hosp, MIMIC-IV-ED/ed and MIMIC-CXR-JPG.
--
-- Unit of analysis : adult ED stay with at least one frontal (PA/AP) chest radiograph during the stay.
-- Prediction time  : t0 = acquisition time of the first frontal radiograph of the stay.
-- Inputs (<= t0)   : index radiograph, triage chief complaint, demographics, triage vitals and acuity,
--                    laboratory results *available* by t0, comorbidities from earlier admissions.
-- Never inputs     : radiology reports, discharge notes, any diagnosis code (they define the label).
-- Label            : pneumonia = ICD-10 J12-J18 or ICD-9 480-486 on the ED stay or linked admission.
--                    label_report (CheXpert label of the index study) is kept for the sensitivity analysis.

CREATE OR REPLACE VIEW patients   AS SELECT * FROM read_csv_auto('$HOSP/patients.csv.gz');
CREATE OR REPLACE VIEW admissions AS SELECT * FROM read_csv_auto('$HOSP/admissions.csv.gz');
CREATE OR REPLACE VIEW hosp_dx    AS SELECT * FROM read_csv_auto('$HOSP/diagnoses_icd.csv.gz', types={'icd_code': 'VARCHAR'});
CREATE OR REPLACE VIEW labevents  AS SELECT subject_id, itemid, charttime, storetime, valuenum
                                     FROM read_csv_auto('$HOSP/labevents.csv.gz')
                                     WHERE itemid IN (51301, 51222, 51265, 50912, 51006, 50983, 50971, 50882, 50931, 50813, 50889);
CREATE OR REPLACE VIEW edstays    AS SELECT * FROM read_csv_auto('$ED/edstays.csv.gz');
CREATE OR REPLACE VIEW triage     AS SELECT * FROM read_csv_auto('$ED/triage.csv.gz', types={'pain': 'VARCHAR'});
CREATE OR REPLACE VIEW ed_dx      AS SELECT * FROM read_csv_auto('$ED/diagnosis.csv.gz', types={'icd_code': 'VARCHAR'});
CREATE OR REPLACE VIEW cxr_meta   AS SELECT * FROM read_csv_auto('$CXR/mimic-cxr-2.0.0-metadata.csv.gz',
                                                   types={'StudyDate': 'VARCHAR', 'StudyTime': 'DOUBLE'});
CREATE OR REPLACE VIEW chexpert   AS SELECT * FROM read_csv_auto('$CXR/mimic-cxr-2.0.0-chexpert.csv.gz');

WITH
cxr AS (
    -- StudyDate is YYYYMMDD and StudyTime is HHMMSS.fff, date-shifted consistently with MIMIC-IV.
    SELECT dicom_id, subject_id, study_id, ViewPosition AS view_position,
           strptime(CAST(StudyDate AS VARCHAR) || lpad(CAST(CAST(floor(StudyTime) AS BIGINT) AS VARCHAR), 6, '0'),
                    '%Y%m%d%H%M%S') AS study_time
    FROM cxr_meta
    WHERE ViewPosition IN ('PA', 'AP')
),
stay_cxr AS (
    SELECT s.stay_id, s.subject_id, s.hadm_id, s.intime, s.outtime, s.gender,
           c.dicom_id, c.study_id, c.view_position, c.study_time,
           ROW_NUMBER() OVER (PARTITION BY s.stay_id
                              ORDER BY c.study_time, (c.view_position = 'PA') DESC, c.dicom_id) AS rn
    FROM edstays s
    JOIN cxr c ON c.subject_id = s.subject_id AND c.study_time BETWEEN s.intime AND s.outtime
),
index_cxr AS (
    SELECT stay_id, subject_id, hadm_id, intime, outtime, gender, dicom_id, study_id, view_position,
           study_time AS t0
    FROM stay_cxr WHERE rn = 1
),
adult AS (
    SELECT i.*, p.anchor_age + (year(i.intime) - p.anchor_year) AS age
    FROM index_cxr i
    JOIN patients p USING (subject_id)
    WHERE p.anchor_age + (year(i.intime) - p.anchor_year) >= 18
),
labs AS (
    -- Latest value per test whose result was *available* (storetime, else charttime) by t0,
    -- restricted to the current ED stay. Item ids: check against d_labitems for your release.
    SELECT a.stay_id,
        arg_max(l.valuenum, l.charttime) FILTER (WHERE l.itemid = 51301) AS wbc,
        arg_max(l.valuenum, l.charttime) FILTER (WHERE l.itemid = 51222) AS hemoglobin,
        arg_max(l.valuenum, l.charttime) FILTER (WHERE l.itemid = 51265) AS platelets,
        arg_max(l.valuenum, l.charttime) FILTER (WHERE l.itemid = 50912) AS creatinine,
        arg_max(l.valuenum, l.charttime) FILTER (WHERE l.itemid = 51006) AS bun,
        arg_max(l.valuenum, l.charttime) FILTER (WHERE l.itemid = 50983) AS sodium,
        arg_max(l.valuenum, l.charttime) FILTER (WHERE l.itemid = 50971) AS potassium,
        arg_max(l.valuenum, l.charttime) FILTER (WHERE l.itemid = 50882) AS bicarbonate,
        arg_max(l.valuenum, l.charttime) FILTER (WHERE l.itemid = 50931) AS glucose,
        arg_max(l.valuenum, l.charttime) FILTER (WHERE l.itemid = 50813) AS lactate,
        arg_max(l.valuenum, l.charttime) FILTER (WHERE l.itemid = 50889) AS crp
    FROM adult a
    JOIN labevents l
      ON l.subject_id = a.subject_id
     AND l.charttime >= a.intime
     AND COALESCE(l.storetime, l.charttime) <= a.t0
     AND l.valuenum IS NOT NULL
    GROUP BY a.stay_id
),
history AS (
    -- Comorbidities coded on admissions discharged before this ED arrival.
    SELECT a.stay_id,
        MAX(CASE WHEN (d.icd_version = 10 AND d.icd_code LIKE 'J44%')
                   OR (d.icd_version = 9 AND (d.icd_code LIKE '491%' OR d.icd_code LIKE '492%' OR d.icd_code LIKE '496%'))
                 THEN 1 ELSE 0 END) AS hx_copd,
        MAX(CASE WHEN (d.icd_version = 10 AND d.icd_code LIKE 'I50%') OR (d.icd_version = 9 AND d.icd_code LIKE '428%')
                 THEN 1 ELSE 0 END) AS hx_chf,
        MAX(CASE WHEN (d.icd_version = 10 AND substr(d.icd_code, 1, 3) BETWEEN 'E08' AND 'E13')
                   OR (d.icd_version = 9 AND d.icd_code LIKE '250%')
                 THEN 1 ELSE 0 END) AS hx_diabetes,
        MAX(CASE WHEN (d.icd_version = 10 AND d.icd_code LIKE 'N18%') OR (d.icd_version = 9 AND d.icd_code LIKE '585%')
                 THEN 1 ELSE 0 END) AS hx_ckd,
        MAX(CASE WHEN (d.icd_version = 10 AND d.icd_code LIKE 'J45%') OR (d.icd_version = 9 AND d.icd_code LIKE '493%')
                 THEN 1 ELSE 0 END) AS hx_asthma,
        MAX(CASE WHEN (d.icd_version = 10 AND substr(d.icd_code, 1, 3) BETWEEN 'J12' AND 'J18')
                   OR (d.icd_version = 9 AND substr(d.icd_code, 1, 3) BETWEEN '480' AND '486')
                 THEN 1 ELSE 0 END) AS hx_pneumonia
    FROM adult a
    JOIN admissions ad ON ad.subject_id = a.subject_id AND ad.dischtime < a.intime
    JOIN hosp_dx d ON d.hadm_id = ad.hadm_id
    GROUP BY a.stay_id
),
pneumonia AS (
    SELECT DISTINCT stay_id FROM ed_dx
    WHERE (icd_version = 10 AND substr(icd_code, 1, 3) BETWEEN 'J12' AND 'J18')
       OR (icd_version = 9 AND substr(icd_code, 1, 3) BETWEEN '480' AND '486')
    UNION
    SELECT DISTINCT a.stay_id FROM adult a
    JOIN hosp_dx d ON d.hadm_id = a.hadm_id
    WHERE (d.icd_version = 10 AND substr(d.icd_code, 1, 3) BETWEEN 'J12' AND 'J18')
       OR (d.icd_version = 9 AND substr(d.icd_code, 1, 3) BETWEEN '480' AND '486')
)
SELECT
    a.stay_id, a.subject_id, a.hadm_id, a.intime, a.t0, a.study_id, a.dicom_id, a.view_position,
    'files/p' || substr(CAST(a.subject_id AS VARCHAR), 1, 2) || '/p' || CAST(a.subject_id AS VARCHAR)
        || '/s' || CAST(a.study_id AS VARCHAR) || '/' || a.dicom_id || '.jpg' AS image_path,
    t.chiefcomplaint AS chief_complaint,
    a.age, a.gender AS sex, t.acuity,
    t.temperature, t.heartrate, t.resprate, t.o2sat, t.sbp, t.dbp, TRY_CAST(t.pain AS DOUBLE) AS pain,
    l.wbc, l.hemoglobin, l.platelets, l.creatinine, l.bun, l.sodium, l.potassium, l.bicarbonate,
    l.glucose, l.lactate, l.crp,
    COALESCE(h.hx_copd, 0) AS hx_copd, COALESCE(h.hx_chf, 0) AS hx_chf,
    COALESCE(h.hx_diabetes, 0) AS hx_diabetes, COALESCE(h.hx_ckd, 0) AS hx_ckd,
    COALESCE(h.hx_asthma, 0) AS hx_asthma, COALESCE(h.hx_pneumonia, 0) AS hx_pneumonia,
    CASE WHEN pn.stay_id IS NOT NULL THEN 1 ELSE 0 END AS label,
    CASE WHEN cx.Pneumonia = 1 THEN 1 WHEN cx.Pneumonia = 0 OR cx.Pneumonia IS NULL THEN 0 END AS label_report
FROM adult a
LEFT JOIN triage t     ON t.stay_id = a.stay_id
LEFT JOIN labs l       ON l.stay_id = a.stay_id
LEFT JOIN history h    ON h.stay_id = a.stay_id
LEFT JOIN pneumonia pn ON pn.stay_id = a.stay_id
LEFT JOIN chexpert cx  ON cx.subject_id = a.subject_id AND cx.study_id = a.study_id
ORDER BY a.subject_id, a.intime
