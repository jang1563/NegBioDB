-- NegBioDB Clinical Trial Failure Domain — Schema Fixes
-- Migration 002: Expert review fixes (6 issues)
--
-- Issues addressed:
--   0.1 UNIQUE constraint on trial_failure_results (dedup protection)
--   0.2 highest_phase_reached ordering (fixed in ct_db.py, not SQL)
--   0.3 inchikey + inchikey_connectivity on interventions (DTI bridge)
--   0.4 molecular_type on interventions (ML featurization)
--   0.5 result_interpretation on trial_failure_results (neg vs inconclusive)
--   0.6 termination_type on clinical_trials (admin vs clinical failure)

-- 0.1: Dedup protection — prevent same failure record from same source
CREATE UNIQUE INDEX IF NOT EXISTS idx_tfr_unique_source
    ON trial_failure_results(
        intervention_id, condition_id,
        COALESCE(trial_id, -1),
        source_db, source_record_id);

-- 0.3: InChIKey columns for DTI-CT cross-domain bridge
ALTER TABLE interventions ADD COLUMN inchikey TEXT;
ALTER TABLE interventions ADD COLUMN inchikey_connectivity TEXT;

-- 0.4: Molecular type for ML featurization (biologics need different features)
ALTER TABLE interventions ADD COLUMN molecular_type TEXT CHECK(molecular_type IN (
    'small_molecule', 'monoclonal_antibody', 'antibody_drug_conjugate',
    'peptide', 'oligonucleotide', 'cell_therapy', 'gene_therapy',
    'other_biologic', 'unknown'));

-- 0.5: Result interpretation (definitive negative vs inconclusive)
ALTER TABLE trial_failure_results ADD COLUMN result_interpretation TEXT CHECK(
    result_interpretation IN (
        'definitive_negative', 'inconclusive_underpowered',
        'mixed_endpoints', 'futility_stopped',
        'safety_stopped', 'administrative'));

-- 0.6: Termination type (clinical failure vs administrative)
ALTER TABLE clinical_trials ADD COLUMN termination_type TEXT CHECK(
    termination_type IN (
        'clinical_failure', 'administrative',
        'external_event', 'unknown'));

-- Index on new columns used in queries
CREATE INDEX IF NOT EXISTS idx_interv_inchikey ON interventions(inchikey_connectivity);
CREATE INDEX IF NOT EXISTS idx_interv_moltype ON interventions(molecular_type);
CREATE INDEX IF NOT EXISTS idx_ct_termtype ON clinical_trials(termination_type);

-- Record migration
INSERT INTO schema_migrations (version) VALUES ('002');
