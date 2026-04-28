# Tesla SQL Cheatsheet

Generated: April 27, 2026

Purpose: a practical SQL cheatsheet for Tesla ML Engineer technical screens. Covers the patterns that appear most often in Tesla data interviews — joins, window functions, deduplication, rolling metrics, and ETL grain checks — all illustrated with Tesla-flavored schemas.

Typical screen format: 10–20 minutes, one or two medium-complexity problems, often using schemas named `drive_events`, `routes`, `fleet_logs`, `collision_events`, or `calibration_metrics`.

---

## Interview Mindset

Say the grain out loud before you write a single line.

```text
Before typing anything, ask or state:
1. Table grain     — one row per what? (event, session, device-day, trip?)
2. Output grain    — what does one row in my result mean?
3. Time filter     — which column? UTC or local? inclusive/exclusive bounds?
4. NULL handling   — should NULLs be counted, excluded, or replaced?
5. Dedup convention — if duplicates exist, keep first? keep latest? keep max?
6. Top-K scope     — top-K globally, or top-K per partition (region, model, day)?
```

Default pattern:

```text
1. Identify grain of each table (comment it above the CTE).
2. Write skeleton: FROM + JOIN + WHERE + GROUP BY + HAVING + ORDER BY.
3. Add window functions in a separate CTE so aggregations don't collide.
4. Sanity check: does row count make sense? Could JOIN fan out?
5. Handle NULLs explicitly — never silently drop them.
```

---

## SQL Cheatsheet

### Basic SELECT / WHERE / GROUP BY / HAVING

```sql
-- drive_events grain: one row per (vehicle_id, event_id)
SELECT
    vehicle_id,
    event_type,
    COUNT(*)                         AS event_count,   -- total rows per group
    COUNT(DISTINCT session_id)       AS unique_sessions,
    AVG(duration_sec)                AS avg_duration
FROM drive_events
WHERE
    occurred_at >= '2024-01-01'      -- time filter on indexed column
    AND event_type != 'IDLE'         -- exclude irrelevant rows
GROUP BY
    vehicle_id,
    event_type
HAVING
    COUNT(*) > 10                    -- filter on aggregated result (post-GROUP BY)
ORDER BY
    avg_duration DESC
LIMIT 50;
```

Key rules:

- Every column in SELECT that is not an aggregate must appear in GROUP BY.
- WHERE runs before aggregation; HAVING runs after. Never put an aggregate in WHERE.
- Use LIMIT + ORDER BY together, otherwise the "top N" is arbitrary.

---

### JOIN Types

```sql
-- INNER JOIN — keep only rows that match in both tables
SELECT de.vehicle_id, r.route_name
FROM drive_events  AS de
INNER JOIN routes  AS r  ON de.route_id = r.route_id;
-- Use when: you only care about events with a known route.

-- LEFT JOIN — keep all left rows; NULLs on right when no match
SELECT de.vehicle_id, r.route_name
FROM drive_events  AS de
LEFT JOIN routes   AS r  ON de.route_id = r.route_id;
-- Use when: you want all events, even those without a route record.

-- RIGHT JOIN — keep all right rows; rarely used; swap table order and use LEFT
SELECT de.vehicle_id, r.route_name
FROM routes        AS r
RIGHT JOIN drive_events AS de ON de.route_id = r.route_id;

-- FULL OUTER JOIN — keep all rows from both sides
SELECT COALESCE(de.vehicle_id, r.route_id) AS id
FROM drive_events  AS de
FULL OUTER JOIN routes AS r ON de.route_id = r.route_id;
-- Use when: auditing mismatches; finding orphans on either side.

-- CROSS JOIN — cartesian product, every combination
SELECT m.model_version, s.scenario_type
FROM models     AS m
CROSS JOIN scenarios AS s;
-- Use when: generating all (model, scenario) pairs for a coverage matrix.

-- SELF JOIN — join a table to itself
SELECT a.vehicle_id, a.occurred_at AS t1, b.occurred_at AS t2
FROM drive_events AS a
JOIN drive_events AS b
  ON  a.vehicle_id = b.vehicle_id
  AND b.occurred_at > a.occurred_at   -- b is a later event for same vehicle
  AND b.occurred_at < a.occurred_at + INTERVAL '5 minutes';
-- Use when: finding pairs of events within a time window for the same entity.
```

JOIN fan-out warning: if the right table has duplicate keys, each left row multiplies. Always check uniqueness before joining on a non-PK column.

---

### Aggregations

```sql
SELECT
    region,
    COUNT(*)                                    AS total_events,
    COUNT(event_id)                             AS non_null_events, -- ignores NULL event_id
    COUNT(DISTINCT vehicle_id)                  AS unique_vehicles,
    SUM(collision_flag)                         AS total_collisions,
    AVG(speed_mph)                              AS avg_speed,
    MIN(occurred_at)                            AS first_event,
    MAX(occurred_at)                            AS last_event,
    SUM(collision_flag) FILTER (WHERE scenario = 'HIGHWAY') AS highway_collisions,
    -- FILTER clause is cleaner than wrapping a CASE WHEN inside SUM
    ROUND(
        100.0 * SUM(collision_flag) / NULLIF(COUNT(*), 0), 2
    )                                           AS collision_rate_pct
FROM drive_events
GROUP BY region;
```

Notes:

- `COUNT(*)` counts rows including NULL values; `COUNT(col)` skips NULLs.
- `NULLIF(expr, 0)` prevents division-by-zero by returning NULL instead of error.
- `FILTER (WHERE ...)` applies a predicate inside an aggregate without a subquery.

---

### Window Functions

```sql
-- grain: one row per (vehicle_id, session_id, occurred_at)
SELECT
    vehicle_id,
    session_id,
    occurred_at,
    speed_mph,

    -- Ranking within each vehicle's session history
    ROW_NUMBER() OVER (
        PARTITION BY vehicle_id
        ORDER BY occurred_at DESC          -- 1 = most recent event per vehicle
    )                                       AS rn,

    RANK() OVER (
        PARTITION BY region
        ORDER BY collision_count DESC       -- ties get same rank, next rank skips
    )                                       AS collision_rank,

    DENSE_RANK() OVER (
        PARTITION BY region
        ORDER BY collision_count DESC       -- ties get same rank, next rank does NOT skip
    )                                       AS collision_dense_rank,

    -- Access adjacent rows
    LAG(speed_mph,  1) OVER (
        PARTITION BY vehicle_id
        ORDER BY occurred_at               -- previous row's speed for same vehicle
    )                                       AS prev_speed,

    LEAD(occurred_at, 1) OVER (
        PARTITION BY vehicle_id
        ORDER BY occurred_at               -- next event's timestamp
    )                                       AS next_event_time,

    -- Rolling aggregations
    SUM(collision_flag) OVER (
        PARTITION BY vehicle_id
        ORDER BY occurred_at
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW   -- 7-row rolling window
    )                                       AS rolling_7_collisions,

    AVG(speed_mph) OVER (
        PARTITION BY vehicle_id
        ORDER BY occurred_at
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW  -- 30-row rolling average
    )                                       AS rolling_30_avg_speed

FROM drive_events;
```

Window frame defaults:

- `ORDER BY` without a frame clause defaults to `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.
- Use `ROWS BETWEEN` for numeric rolling windows; `RANGE BETWEEN` for value-based ranges.
- Window functions run after WHERE and GROUP BY but before the outer SELECT.

---

### CTEs and Recursive CTEs

```sql
-- Non-recursive CTE: break a complex query into named steps
WITH
daily_collisions AS (
    -- grain: one row per (region, event_date)
    SELECT
        region,
        DATE_TRUNC('day', occurred_at)      AS event_date,
        COUNT(*) FILTER (WHERE collision_flag = 1) AS collisions,
        COUNT(*)                            AS total_events
    FROM drive_events
    GROUP BY 1, 2
),
collision_rates AS (
    -- grain: one row per (region, event_date)
    SELECT
        region,
        event_date,
        collisions,
        total_events,
        ROUND(collisions * 1.0 / NULLIF(total_events, 0), 4) AS collision_rate
    FROM daily_collisions
),
ranked AS (
    -- rank regions by collision rate per day
    SELECT
        *,
        RANK() OVER (PARTITION BY event_date ORDER BY collision_rate DESC) AS day_rank
    FROM collision_rates
)
SELECT * FROM ranked WHERE day_rank <= 3;  -- top 3 regions per day
```

```sql
-- Recursive CTE: traverse hierarchical or sequence data
WITH RECURSIVE session_chain AS (
    -- Base case: seed with first event of each session
    SELECT vehicle_id, session_id, event_id, occurred_at, 1 AS depth
    FROM drive_events
    WHERE prev_event_id IS NULL             -- first event has no predecessor

    UNION ALL

    -- Recursive case: follow chain forward
    SELECT de.vehicle_id, de.session_id, de.event_id, de.occurred_at, sc.depth + 1
    FROM drive_events        AS de
    JOIN session_chain       AS sc
      ON de.prev_event_id = sc.event_id
    WHERE sc.depth < 100                    -- guard against infinite loops
)
SELECT vehicle_id, session_id, MAX(depth) AS chain_length
FROM session_chain
GROUP BY 1, 2;
```

When to use CTEs vs subqueries vs JOINs:

```text
CTE         — logic is reused, or the query has 3+ logical stages that help readability.
Subquery    — one-off filter or scalar lookup; keep it inline to avoid naming overhead.
JOIN        — you need columns from both sides in the SELECT list.
             Avoid correlated subqueries in WHERE if a JOIN can do the same job faster.
```

---

### Date and Time Functions

```sql
-- Truncate to day / hour / week / month
DATE_TRUNC('day',   occurred_at)   -- 2024-03-15 14:23:00 -> 2024-03-15 00:00:00
DATE_TRUNC('hour',  occurred_at)   -- -> 2024-03-15 14:00:00
DATE_TRUNC('week',  occurred_at)   -- -> week-starting Monday
DATE_TRUNC('month', occurred_at)   -- -> 2024-03-01 00:00:00

-- Difference between timestamps
DATEDIFF('day',  trip_start, trip_end)    -- Snowflake / Redshift syntax
DATE_PART('epoch', trip_end - trip_start) -- Postgres: seconds between

-- Extract a component
EXTRACT(DOW  FROM occurred_at)   -- day of week: 0=Sunday, 6=Saturday
EXTRACT(HOUR FROM occurred_at)   -- 0-23
EXTRACT(WEEK FROM occurred_at)   -- ISO week number

-- Time bucketing (5-minute buckets for real-time dashboards)
DATE_TRUNC('minute', occurred_at)
  - MAKE_INTERVAL(mins := EXTRACT(MINUTE FROM occurred_at)::INT % 5)
-- Simpler in Snowflake:
TIME_SLICE(occurred_at, 5, 'MINUTE')

-- Add/subtract intervals
occurred_at + INTERVAL '7 days'
occurred_at - INTERVAL '1 hour'

-- Convert time zone
occurred_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles'
```

Time zone pitfall: Tesla telemetry is stored in UTC. Always convert to local time only in the outermost SELECT for display; filter and aggregate in UTC.

---

### String Functions

```sql
-- Pattern matching
WHERE event_type LIKE  'COLLISION%'        -- prefix match; % = any sequence
WHERE event_type LIKE  '%NEAR_MISS%'       -- substring match
WHERE event_type ILIKE 'collision%'        -- case-insensitive (Postgres/Snowflake)
WHERE vin        REGEXP '^5YJ[0-9A-Z]{14}$' -- regex match for Tesla VIN format

-- Extract parts of strings
SUBSTRING(vin, 1, 3)                      -- first 3 chars: manufacturer code
SPLIT_PART(scenario_tag, '::', 2)         -- second element of '::'-delimited tag
LEFT(model_version, 7)                    -- first 7 chars
UPPER(region_code)
LOWER(event_type)
TRIM(BOTH ' ' FROM description)

-- Build strings
CONCAT(vehicle_id, '-', session_id)
vehicle_id || '-' || session_id           -- Postgres string concat
```

---

### NULL Handling

```sql
-- Replace NULL with a default
COALESCE(calibration_score, 0)             -- first non-NULL in the list
COALESCE(city, state, region, 'UNKNOWN')   -- chain of fallbacks

-- Produce NULL on a sentinel value (prevents divide-by-zero)
NULLIF(total_events, 0)                    -- returns NULL when denominator is 0
NULLIF(label, '')                          -- treat empty string as NULL

-- Filter on NULL
WHERE calibration_score IS NULL            -- no calibration record
WHERE calibration_score IS NOT NULL        -- has a record

-- NULL-safe equality (Snowflake / BigQuery)
WHERE a IS NOT DISTINCT FROM b             -- true if both NULL, or both equal
-- Standard SQL equivalent:
WHERE (a = b) OR (a IS NULL AND b IS NULL)

-- NULL propagation rules
NULL + 5          -> NULL    -- any arithmetic with NULL is NULL
NULL = NULL       -> NULL    -- not TRUE; use IS NULL instead
COUNT(NULL_col)   -> 0       -- COUNT skips NULLs
SUM(NULL_col)     -> NULL    -- SUM of all-NULLs is NULL, not 0
AVG(NULL_col)     -> NULL    -- AVG skips NULLs in numerator and denominator
```

---

## Top Asked Questions

### Q1. Top-K routes by collision count per day

Tesla question: "Given `drive_events(event_id, vehicle_id, route_id, occurred_at, collision_flag)` and `routes(route_id, route_name, region)`, return the top 3 routes by collision count for each calendar day."

```sql
WITH daily_route_collisions AS (
    -- grain: one row per (event_date, route_id)
    SELECT
        DATE_TRUNC('day', de.occurred_at) AS event_date,
        de.route_id,
        r.route_name,
        SUM(de.collision_flag)            AS collision_count
    FROM drive_events AS de
    INNER JOIN routes AS r ON de.route_id = r.route_id
    WHERE de.collision_flag = 1           -- only collision events
    GROUP BY 1, 2, 3
),
ranked AS (
    SELECT
        *,
        RANK() OVER (
            PARTITION BY event_date
            ORDER BY collision_count DESC  -- highest collisions ranked 1
        ) AS day_rank
    FROM daily_route_collisions
)
SELECT event_date, route_name, collision_count, day_rank
FROM ranked
WHERE day_rank <= 3                        -- top 3 per day
ORDER BY event_date, day_rank;
```

Key choices: RANK vs ROW_NUMBER. RANK gives ties the same position; ROW_NUMBER breaks ties arbitrarily. Clarify with the interviewer.

---

### Q2. Deduplicate fleet logs — keep latest per device_id

Tesla question: "`fleet_logs(log_id, device_id, firmware_version, logged_at, status)` has duplicate rows per device. Return the most recent log per device."

```sql
WITH latest_per_device AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY device_id
            ORDER BY logged_at DESC        -- 1 = most recent log
        ) AS rn
    FROM fleet_logs
)
SELECT log_id, device_id, firmware_version, logged_at, status
FROM latest_per_device
WHERE rn = 1;                             -- keep only the latest row
```

Alternative using a self-JOIN (useful when window functions are not available):

```sql
SELECT fl.*
FROM fleet_logs AS fl
INNER JOIN (
    SELECT device_id, MAX(logged_at) AS max_logged_at
    FROM fleet_logs
    GROUP BY device_id
) AS latest
  ON  fl.device_id    = latest.device_id
  AND fl.logged_at    = latest.max_logged_at;
-- Risk: if two rows have the exact same max timestamp, this returns both.
-- ROW_NUMBER approach is safer and more explicit.
```

---

### Q3. 7-day rolling collision rate per region

Tesla question: "Compute the 7-day rolling collision rate (collisions / total events) for each region, using `drive_events`."

```sql
WITH daily_region AS (
    -- grain: one row per (region, event_date)
    SELECT
        region,
        DATE_TRUNC('day', occurred_at)           AS event_date,
        SUM(collision_flag)                       AS daily_collisions,
        COUNT(*)                                  AS daily_events
    FROM drive_events
    GROUP BY 1, 2
),
rolling AS (
    SELECT
        region,
        event_date,
        SUM(daily_collisions) OVER (
            PARTITION BY region
            ORDER BY event_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW  -- 7 days total
        )                                        AS rolling_7d_collisions,
        SUM(daily_events) OVER (
            PARTITION BY region
            ORDER BY event_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )                                        AS rolling_7d_events
    FROM daily_region
)
SELECT
    region,
    event_date,
    rolling_7d_collisions,
    rolling_7d_events,
    ROUND(
        rolling_7d_collisions * 1.0 / NULLIF(rolling_7d_events, 0), 5
    )                                            AS rolling_7d_collision_rate
FROM rolling
ORDER BY region, event_date;
```

---

### Q4. Sessions with no calibration record (anti-join)

Tesla question: "Find all drive sessions that have no associated calibration record. Tables: `drive_sessions(session_id, vehicle_id, started_at)`, `calibration_metrics(session_id, calibration_score, calibrated_at)`."

```sql
-- Method 1: LEFT JOIN + IS NULL (most common in interviews)
SELECT ds.session_id, ds.vehicle_id, ds.started_at
FROM drive_sessions     AS ds
LEFT JOIN calibration_metrics AS cm ON ds.session_id = cm.session_id
WHERE cm.session_id IS NULL;             -- NULL means no matching calibration row

-- Method 2: NOT EXISTS (explicit intent, cleaner logic)
SELECT session_id, vehicle_id, started_at
FROM drive_sessions AS ds
WHERE NOT EXISTS (
    SELECT 1
    FROM calibration_metrics AS cm
    WHERE cm.session_id = ds.session_id
);

-- Method 3: NOT IN (avoid if cm.session_id can be NULL — NOT IN with NULLs returns 0 rows)
SELECT session_id, vehicle_id, started_at
FROM drive_sessions
WHERE session_id NOT IN (
    SELECT session_id FROM calibration_metrics WHERE session_id IS NOT NULL
);
```

Prefer LEFT JOIN + IS NULL or NOT EXISTS. NOT IN silently fails when the subquery returns any NULL.

---

### Q5. Compute precision and recall from a confusion matrix table

Tesla question: "Table `confusion_matrix(model_version, label, predicted, count)` contains TP/FP/FN/TN counts. Compute precision and recall per model version."

```sql
WITH counts AS (
    SELECT
        model_version,
        SUM(count) FILTER (WHERE label = 1 AND predicted = 1) AS tp,
        SUM(count) FILTER (WHERE label = 0 AND predicted = 1) AS fp,
        SUM(count) FILTER (WHERE label = 1 AND predicted = 0) AS fn,
        SUM(count) FILTER (WHERE label = 0 AND predicted = 0) AS tn
    FROM confusion_matrix
    GROUP BY model_version
)
SELECT
    model_version,
    tp, fp, fn, tn,
    ROUND(tp * 1.0 / NULLIF(tp + fp, 0), 4) AS precision,   -- of predicted positive, how many are correct
    ROUND(tp * 1.0 / NULLIF(tp + fn, 0), 4) AS recall,      -- of actual positive, how many did we catch
    ROUND(
        2.0 * tp / NULLIF(2 * tp + fp + fn, 0), 4
    )                                        AS f1_score
FROM counts
ORDER BY model_version;
```

---

### Q6. Second-highest calibration score per model version

Tesla question: "Find the second-highest calibration score for each model version in `calibration_metrics(run_id, model_version, calibration_score, run_date)`."

```sql
WITH ranked AS (
    SELECT
        run_id,
        model_version,
        calibration_score,
        run_date,
        DENSE_RANK() OVER (
            PARTITION BY model_version
            ORDER BY calibration_score DESC  -- 1 = highest score
        ) AS score_rank
    FROM calibration_metrics
    WHERE calibration_score IS NOT NULL      -- exclude NULL scores from ranking
)
SELECT model_version, calibration_score, run_date
FROM ranked
WHERE score_rank = 2;                        -- second-highest per model version
```

Why DENSE_RANK: if multiple runs share the highest score, DENSE_RANK still assigns rank 2 to the next distinct score, while RANK would skip to 3.

---

### Q7. Detect gaps in time-series telemetry data

Tesla question: "Each vehicle should emit a heartbeat event every 5 minutes. Using `heartbeats(heartbeat_id, vehicle_id, emitted_at)`, find all gaps longer than 10 minutes for each vehicle."

```sql
WITH ordered AS (
    -- grain: one row per heartbeat, ordered per vehicle
    SELECT
        vehicle_id,
        emitted_at,
        LEAD(emitted_at) OVER (
            PARTITION BY vehicle_id
            ORDER BY emitted_at         -- next heartbeat timestamp for this vehicle
        ) AS next_emitted_at
    FROM heartbeats
)
SELECT
    vehicle_id,
    emitted_at                          AS gap_start,
    next_emitted_at                     AS gap_end,
    EXTRACT(EPOCH FROM (next_emitted_at - emitted_at)) / 60 AS gap_minutes
FROM ordered
WHERE next_emitted_at IS NOT NULL        -- skip the last row per vehicle (no successor)
  AND next_emitted_at - emitted_at > INTERVAL '10 minutes'
ORDER BY vehicle_id, gap_start;
```

---

### Q8. Pivot a long table into wide (CASE WHEN aggregation)

Tesla question: "`drive_events(vehicle_id, event_type, occurred_at)` stores event types as rows. Pivot to one row per vehicle with columns for each event type count: COLLISION, NEAR_MISS, LANE_CHANGE."

```sql
SELECT
    vehicle_id,
    COUNT(*) FILTER (WHERE event_type = 'COLLISION')    AS collision_count,
    COUNT(*) FILTER (WHERE event_type = 'NEAR_MISS')    AS near_miss_count,
    COUNT(*) FILTER (WHERE event_type = 'LANE_CHANGE')  AS lane_change_count,
    COUNT(*)                                             AS total_events
FROM drive_events
GROUP BY vehicle_id
ORDER BY collision_count DESC;

-- Alternative using CASE WHEN (works in databases without FILTER):
SELECT
    vehicle_id,
    SUM(CASE WHEN event_type = 'COLLISION'   THEN 1 ELSE 0 END) AS collision_count,
    SUM(CASE WHEN event_type = 'NEAR_MISS'   THEN 1 ELSE 0 END) AS near_miss_count,
    SUM(CASE WHEN event_type = 'LANE_CHANGE' THEN 1 ELSE 0 END) AS lane_change_count
FROM drive_events
GROUP BY vehicle_id;
```

---

### Q9. Find vehicles with consecutive collision events

Tesla question: "A vehicle is flagged as 'high-risk' if it has 3 or more collision events in a row (ordered by `occurred_at`, with no non-collision event in between). Find all high-risk vehicles."

```sql
-- Gaps-and-islands trick: assign an "island_id" so consecutive same-type rows share an id.
-- The classic form: rn_overall - rn_within_type is constant across a run of the same type.
WITH numbered AS (
    SELECT
        vehicle_id,
        event_type,
        occurred_at,
        ROW_NUMBER() OVER (
            PARTITION BY vehicle_id ORDER BY occurred_at
        ) AS rn_overall,                -- row number across ALL events for the vehicle
        ROW_NUMBER() OVER (
            PARTITION BY vehicle_id, event_type ORDER BY occurred_at
        ) AS rn_per_type                -- row number within (vehicle, event_type)
    FROM drive_events                   -- IMPORTANT: do NOT pre-filter to COLLISION; we need other types to break runs
),
islands AS (
    SELECT
        vehicle_id,
        event_type,
        rn_overall - rn_per_type AS island_id   -- constant within a run of the same event_type
    FROM numbered
),
island_sizes AS (
    SELECT
        vehicle_id,
        event_type,
        island_id,
        COUNT(*) AS run_length
    FROM islands
    GROUP BY vehicle_id, event_type, island_id
)
SELECT DISTINCT vehicle_id
FROM island_sizes
WHERE event_type = 'COLLISION'
  AND run_length >= 3;
```

Why we cannot pre-filter to COLLISION: the row-number-difference trick relies on seeing the interleaving of types. If we drop non-collisions first, every remaining row is a "consecutive" collision and the result over-counts.

---

### Q10. ETL grain audit — find duplicate primary keys

Tesla question: "After an ETL load, verify that `drive_events` has no duplicate `event_id` values."

```sql
-- Quick count check
SELECT
    COUNT(*)                    AS total_rows,
    COUNT(DISTINCT event_id)    AS unique_event_ids,
    COUNT(*) - COUNT(DISTINCT event_id) AS duplicate_count
FROM drive_events;

-- Find the actual duplicate event_ids
SELECT event_id, COUNT(*) AS occurrences
FROM drive_events
GROUP BY event_id
HAVING COUNT(*) > 1
ORDER BY occurrences DESC;

-- Show all columns of duplicated rows
SELECT de.*
FROM drive_events AS de
INNER JOIN (
    SELECT event_id
    FROM drive_events
    GROUP BY event_id
    HAVING COUNT(*) > 1
) AS dupes ON de.event_id = dupes.event_id
ORDER BY de.event_id, de.occurred_at;
```

---

### Q11. Vehicles that appear in routes but not in fleet registry (symmetric difference)

Tesla question: "Find vehicle_ids that have drive events but no entry in `vehicle_registry(vehicle_id, vin, model, manufactured_at)`, and vice versa."

```sql
-- In drive_events but not in registry
SELECT DISTINCT vehicle_id, 'in_events_only' AS source
FROM drive_events
WHERE vehicle_id NOT IN (SELECT vehicle_id FROM vehicle_registry)

UNION ALL

-- In registry but no drive events recorded
SELECT DISTINCT vehicle_id, 'in_registry_only' AS source
FROM vehicle_registry
WHERE vehicle_id NOT IN (
    SELECT DISTINCT vehicle_id FROM drive_events WHERE vehicle_id IS NOT NULL
);
```

---

### Q12. Compute cumulative distance driven per vehicle per month

Tesla question: "`routes(route_id, vehicle_id, distance_km, completed_at)`. Return cumulative distance per vehicle, by month, in chronological order."

```sql
WITH monthly AS (
    -- grain: one row per (vehicle_id, month)
    SELECT
        vehicle_id,
        DATE_TRUNC('month', completed_at) AS month,
        SUM(distance_km)                  AS monthly_km
    FROM routes
    WHERE completed_at IS NOT NULL
    GROUP BY 1, 2
)
SELECT
    vehicle_id,
    month,
    monthly_km,
    SUM(monthly_km) OVER (
        PARTITION BY vehicle_id
        ORDER BY month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  -- running total
    )                                     AS cumulative_km
FROM monthly
ORDER BY vehicle_id, month;
```

---

## Common Pitfalls

- **Non-grouped columns in SELECT**: every column in SELECT must either be in GROUP BY or wrapped in an aggregate. `SELECT vehicle_id, event_type, COUNT(*) FROM t GROUP BY vehicle_id` will error or give wrong results because `event_type` is not grouped.

- **JOIN fan-out from duplicate keys**: if the right-side table has multiple rows per join key, each left row produces multiple output rows, inflating COUNT and SUM. Always verify uniqueness before joining on a non-primary key column.

- **COUNT(*) vs COUNT(col)**: `COUNT(*)` counts every row including those with NULLs. `COUNT(col)` skips NULL values in that column. Use whichever matches intent; confusing them silently produces wrong denominators in rate calculations.

- **HAVING vs WHERE ordering**: WHERE filters rows before grouping; HAVING filters groups after aggregation. Putting an aggregate condition (`HAVING COUNT(*) > 5`) in WHERE causes a syntax error or silent wrong logic.

- **Window frame defaults**: `ORDER BY col` in a window function without an explicit ROWS/RANGE frame defaults to `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which includes all ties at the current value — not just the current row. Use `ROWS BETWEEN` for precise numeric windows.

- **Time zone bugs**: filtering `WHERE occurred_at >= '2024-01-01'` without specifying a zone compares UTC timestamps against a naive literal. In Snowflake or Postgres, cast explicitly: `WHERE occurred_at >= '2024-01-01 00:00:00'::TIMESTAMPTZ AT TIME ZONE 'UTC'`.

- **NULL in JOIN conditions**: `NULL = NULL` evaluates to NULL, not TRUE. Joining on a nullable column silently drops rows where either side is NULL. Use `IS NOT DISTINCT FROM` or pre-filter NULLs in a CTE.

- **Integer division**: `1 / 2` in SQL returns `0`, not `0.5`. Cast at least one operand: `1.0 / 2` or `CAST(count AS FLOAT) / total`.

- **NOT IN with NULLs**: `WHERE id NOT IN (SELECT id FROM t)` returns zero rows if the subquery contains any NULL, because `x NOT IN (..., NULL)` evaluates to NULL (unknown). Always add `WHERE id IS NOT NULL` inside the subquery.

- **Aggregating before joining**: if you need aggregates from two tables, aggregate each table in a CTE first, then join the summaries. Joining raw tables first and then grouping inflates intermediate row counts and slows the query.

---

## Tesla-Specific Patterns

### Collision rate per scenario type

Tesla monitors how collision rate varies across scenario tags (HIGHWAY, URBAN, PARKING, CONSTRUCTION).

```sql
-- grain: one row per scenario, with collision statistics
SELECT
    scenario_type,
    COUNT(*)                                          AS total_events,
    SUM(collision_flag)                               AS collisions,
    ROUND(SUM(collision_flag) * 1.0 / COUNT(*), 5)   AS collision_rate,
    ROUND(AVG(speed_mph), 1)                          AS avg_speed_mph
FROM drive_events
WHERE occurred_at >= DATEADD('month', -3, CURRENT_DATE)  -- last 90 days
GROUP BY scenario_type
ORDER BY collision_rate DESC;
```

---

### ETL grain audit — verify expected row count per partition

After each daily ETL load, check that the load did not double-insert or skip a partition.

```sql
-- Expected: one file per (region, date); audit the loaded table
SELECT
    region,
    DATE_TRUNC('day', occurred_at)  AS event_date,
    COUNT(*)                        AS loaded_rows,
    COUNT(DISTINCT event_id)        AS unique_events,
    -- flag if we see more rows than unique events (duplicates) or much fewer (missing)
    CASE
        WHEN COUNT(*) > COUNT(DISTINCT event_id) THEN 'DUPLICATE_ROWS'
        WHEN COUNT(DISTINCT event_id) < 1000     THEN 'POSSIBLE_UNDERLOAD'
        ELSE 'OK'
    END                             AS load_status
FROM drive_events
WHERE occurred_at >= DATEADD('day', -7, CURRENT_DATE)
GROUP BY 1, 2
HAVING COUNT(*) > COUNT(DISTINCT event_id)         -- duplicates
    OR COUNT(DISTINCT event_id) < 1000             -- under-load
ORDER BY event_date DESC, region;
```

---

### Data-quality check — detect out-of-range sensor values

```sql
-- Find drive events with implausible speed or GPS readings
SELECT
    event_id,
    vehicle_id,
    occurred_at,
    speed_mph,
    latitude,
    longitude,
    CASE
        WHEN speed_mph < 0          THEN 'NEGATIVE_SPEED'
        WHEN speed_mph > 200        THEN 'SPEED_TOO_HIGH'
        WHEN ABS(latitude)  > 90    THEN 'INVALID_LAT'
        WHEN ABS(longitude) > 180   THEN 'INVALID_LON'
        ELSE 'OK'
    END                             AS quality_flag
FROM drive_events
WHERE occurred_at >= CURRENT_DATE - INTERVAL '1 day'
  AND (
      speed_mph < 0 OR speed_mph > 200
      OR ABS(latitude) > 90
      OR ABS(longitude) > 180
  );
```

---

### Calibration drift detection

Flag vehicles whose calibration score has dropped more than 5% from their personal 30-day average.

```sql
WITH vehicle_avg AS (
    -- grain: one row per vehicle, 30-day average calibration score
    SELECT
        vehicle_id,
        AVG(calibration_score) AS avg_score_30d
    FROM calibration_metrics
    WHERE calibrated_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY vehicle_id
),
latest_calibration AS (
    -- grain: one row per vehicle, most recent score
    SELECT DISTINCT ON (vehicle_id)
        vehicle_id,
        calibration_score,
        calibrated_at
    FROM calibration_metrics
    ORDER BY vehicle_id, calibrated_at DESC
)
SELECT
    lc.vehicle_id,
    lc.calibration_score        AS latest_score,
    va.avg_score_30d,
    ROUND(
        (lc.calibration_score - va.avg_score_30d) / NULLIF(va.avg_score_30d, 0), 4
    )                           AS pct_change
FROM latest_calibration AS lc
INNER JOIN vehicle_avg   AS va ON lc.vehicle_id = va.vehicle_id
WHERE lc.calibration_score < va.avg_score_30d * 0.95  -- more than 5% drop
ORDER BY pct_change ASC;                               -- worst drift first
```

---

### Model performance regression check across software versions

Track whether a new firmware release degraded collision detection recall.

```sql
WITH version_metrics AS (
    SELECT
        firmware_version,
        SUM(tp)                                    AS tp,
        SUM(fp)                                    AS fp,
        SUM(fn)                                    AS fn
    FROM model_eval_results
    WHERE eval_date >= CURRENT_DATE - INTERVAL '14 days'
    GROUP BY firmware_version
)
SELECT
    firmware_version,
    ROUND(tp * 1.0 / NULLIF(tp + fn, 0), 4)       AS recall,
    ROUND(tp * 1.0 / NULLIF(tp + fp, 0), 4)       AS precision,
    ROUND(2.0 * tp / NULLIF(2 * tp + fp + fn, 0), 4) AS f1,
    LAG(ROUND(tp * 1.0 / NULLIF(tp + fn, 0), 4)) OVER (
        ORDER BY firmware_version                  -- compare to previous version
    )                                              AS prev_recall,
    ROUND(tp * 1.0 / NULLIF(tp + fn, 0), 4)
    - LAG(ROUND(tp * 1.0 / NULLIF(tp + fn, 0), 4)) OVER (
        ORDER BY firmware_version
    )                                              AS recall_delta
FROM version_metrics
ORDER BY firmware_version;
```
