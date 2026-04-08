-- ============================================================
-- sample.sql
-- Connect to MySQL Sakila schema and query the actor table
-- ============================================================

USE sakila;

-- Select all records from the actor table
SELECT
    actor_id,
    first_name,
    last_name,
    last_update
FROM
    actor
ORDER BY
    actor_id ASC;
