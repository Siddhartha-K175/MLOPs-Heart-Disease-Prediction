-- Heart-disease application database schema (PostgreSQL 15+ / Supabase).
--
-- The objects live in their own Postgres schema (DB_SCHEMA, default
-- heart_health) rather than `public`, so Supabase's auto-generated REST API
-- does not expose these health records. The application creates that schema
-- and runs this file with search_path pointing at it. To load it by hand,
-- paste it into the Supabase SQL Editor after:
--     CREATE SCHEMA IF NOT EXISTS heart_health;
--     SET search_path TO heart_health;
--
-- api.py executes the same file at startup; every statement is idempotent,
-- so running it repeatedly is safe. The clinical CHECK ranges below mirror
-- FEATURE_BOUNDS in src/model.py.

CREATE TABLE IF NOT EXISTS patients (
    patient_id VARCHAR(64)  PRIMARY KEY,
    name       VARCHAR(120) NULL,
    notes      VARCHAR(500) NULL,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_patients_name ON patients (name);

-- Postgres has no ON UPDATE CURRENT_TIMESTAMP; a trigger keeps updated_at fresh.
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

CREATE OR REPLACE TRIGGER trg_patients_updated_at
    BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- One row per scored prediction: the 13 model inputs plus the model output.
-- patient_id stays NULL for anonymous requests (no identity supplied).
CREATE TABLE IF NOT EXISTS predictions (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    patient_id       VARCHAR(64)      NULL
        REFERENCES patients (patient_id) ON DELETE CASCADE,
    age              DOUBLE PRECISION NOT NULL CHECK (age BETWEEN 1 AND 120),
    sex              DOUBLE PRECISION NOT NULL CHECK (sex BETWEEN 0 AND 1),
    cp               DOUBLE PRECISION NOT NULL CHECK (cp BETWEEN 0 AND 4),
    trestbps         DOUBLE PRECISION NOT NULL CHECK (trestbps BETWEEN 50 AND 300),
    chol             DOUBLE PRECISION NOT NULL CHECK (chol BETWEEN 100 AND 600),
    fbs              DOUBLE PRECISION NOT NULL CHECK (fbs BETWEEN 0 AND 1),
    restecg          DOUBLE PRECISION NOT NULL CHECK (restecg BETWEEN 0 AND 2),
    thalach          DOUBLE PRECISION NOT NULL CHECK (thalach BETWEEN 60 AND 220),
    exang            DOUBLE PRECISION NOT NULL CHECK (exang BETWEEN 0 AND 1),
    oldpeak          DOUBLE PRECISION NOT NULL CHECK (oldpeak BETWEEN 0 AND 10),
    slope            DOUBLE PRECISION NOT NULL CHECK (slope BETWEEN 0 AND 3),
    ca               DOUBLE PRECISION NOT NULL CHECK (ca BETWEEN 0 AND 4),
    thal             DOUBLE PRECISION NOT NULL CHECK (thal BETWEEN 0 AND 7),
    prediction       INTEGER          NOT NULL CHECK (prediction IN (0, 1)),
    risk_probability DOUBLE PRECISION NOT NULL CHECK (risk_probability BETWEEN 0 AND 1),
    risk_label       VARCHAR(120)     NOT NULL,
    model_version    VARCHAR(64)      NULL,
    -- When the measurements were taken (entered on the form); created_at is
    -- when the record was saved.
    assessed_on      DATE             NOT NULL DEFAULT CURRENT_DATE,
    created_at       TIMESTAMPTZ      NOT NULL DEFAULT now()
);

-- Databases created before assessed_on existed: backfill it from created_at.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'predictions'
          AND column_name = 'assessed_on'
    ) THEN
        ALTER TABLE predictions ADD COLUMN assessed_on DATE;
        UPDATE predictions SET assessed_on = created_at::date;
        ALTER TABLE predictions
            ALTER COLUMN assessed_on SET DEFAULT CURRENT_DATE,
            ALTER COLUMN assessed_on SET NOT NULL;
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON predictions (patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_assessed_on ON predictions (assessed_on);

-- Sign-in accounts. Admins read every patient record; a patient account is
-- tied to exactly one patients row and sees only that history.
--  * Admins sign in with a username; their password is stored here only as a
--    scrypt hash (see src/auth.py). Create them with
--        uv run python -m src.auth create-admin <username>
--  * Patients register by email through Supabase Auth, which confirms the
--    address and keeps the password; auth_user_id links the Supabase user
--    and username holds the lower-cased email.
CREATE TABLE IF NOT EXISTS accounts (
    username      VARCHAR(254) PRIMARY KEY,
    password_hash VARCHAR(255) NULL,
    role          VARCHAR(16)  NOT NULL CHECK (role IN ('admin', 'patient')),
    patient_id    VARCHAR(64)  NULL UNIQUE
        REFERENCES patients (patient_id) ON DELETE CASCADE,
    email         VARCHAR(254) NULL UNIQUE,
    auth_user_id  UUID         NULL UNIQUE,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    last_login_at TIMESTAMPTZ  NULL,
    CONSTRAINT chk_accounts_patient_link CHECK ((role = 'patient') = (patient_id IS NOT NULL))
);

-- Accounts tables created before email sign-up: add the new columns.
ALTER TABLE accounts ALTER COLUMN username TYPE VARCHAR(254);
ALTER TABLE accounts ALTER COLUMN password_hash DROP NOT NULL;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS email VARCHAR(254) NULL UNIQUE;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS auth_user_id UUID NULL UNIQUE;

-- Every account signs in somehow: a local password or a Supabase Auth user.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_accounts_credentials'
          AND conrelid = to_regclass('accounts')
    ) THEN
        ALTER TABLE accounts ADD CONSTRAINT chk_accounts_credentials
            CHECK (password_hash IS NOT NULL OR auth_user_id IS NOT NULL);
    END IF;
END;
$$;

-- Signed-in sessions. Only a SHA-256 of each bearer token is stored, so a
-- database leak does not hand out live tokens.
CREATE TABLE IF NOT EXISTS sessions (
    token_hash CHAR(64)    PRIMARY KEY,
    username   VARCHAR(254) NOT NULL REFERENCES accounts (username) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);

ALTER TABLE sessions ALTER COLUMN username TYPE VARCHAR(254);

CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions (expires_at);

-- One-time migration from the earlier admin-only tables: keep every admin and
-- its password hash; old sessions are dropped, so admins sign in again.
DO $$
BEGIN
    IF to_regclass('admins') IS NOT NULL THEN
        INSERT INTO accounts (username, password_hash, role, created_at, last_login_at)
        SELECT lower(username), password_hash, 'admin', created_at, last_login_at
        FROM admins
        ON CONFLICT (username) DO NOTHING;
        DROP TABLE IF EXISTS admin_sessions;
        DROP TABLE admins;
    END IF;
END;
$$;

-- Full patient history: every linked prediction with its patient identity,
-- newest first.
CREATE OR REPLACE VIEW patient_history AS
SELECT
    pr.id               AS record_id,
    pt.patient_id,
    pt.name,
    pr.age,
    pr.sex,
    pr.cp,
    pr.trestbps,
    pr.chol,
    pr.fbs,
    pr.restecg,
    pr.thalach,
    pr.exang,
    pr.oldpeak,
    pr.slope,
    pr.ca,
    pr.thal,
    pr.prediction,
    pr.risk_probability,
    pr.risk_label,
    pr.model_version,
    pr.created_at,
    pr.assessed_on
FROM predictions AS pr
JOIN patients AS pt ON pt.patient_id = pr.patient_id
ORDER BY pr.assessed_on DESC, pr.created_at DESC, pr.id DESC;
