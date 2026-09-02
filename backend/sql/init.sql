-- =============================================================================
-- Bootstrap script executed by PostgreSQL's docker-entrypoint-initdb.d on the
-- FIRST boot of the postgres container (only when the data dir is empty).
--
-- It prepares the `app_user` role so that subsequent `alembic upgrade head`
-- commands (which GRANT/REVOKE to that role) succeed on a fresh container.
--
-- The bulk of the schema is created by Alembic migration `0001_initial`,
-- NOT by this file. Keeping the role creation here avoids a chicken-and-egg
-- problem where Alembic needs the role to exist to GRANT privileges to it.
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user LOGIN PASSWORD 'changeme';
    END IF;
END $$;

-- Allow the role to connect to the bootstrap database and use the public
-- schema. Alembic migrations (run by the superuser) refine grants later.
GRANT CONNECT ON DATABASE football TO app_user;
GRANT USAGE, CREATE ON SCHEMA public TO app_user;
