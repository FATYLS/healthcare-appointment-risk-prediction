-- PostgreSQL-oriented analytical queries.
-- Expected table: appointments with engineered columns.

SELECT AVG(target_no_show::int) AS no_show_rate
FROM appointments;

SELECT
    sms_received,
    COUNT(*) AS appointments,
    AVG(target_no_show::int) AS no_show_rate
FROM appointments
GROUP BY sms_received
ORDER BY sms_received;

SELECT
    CASE
        WHEN waiting_days <= 1 THEN '0-1'
        WHEN waiting_days <= 3 THEN '2-3'
        WHEN waiting_days <= 7 THEN '4-7'
        WHEN waiting_days <= 14 THEN '8-14'
        WHEN waiting_days <= 30 THEN '15-30'
        ELSE '31+'
    END AS waiting_bucket,
    COUNT(*) AS appointments,
    AVG(target_no_show::int) AS no_show_rate
FROM appointments
GROUP BY 1
ORDER BY MIN(waiting_days);
