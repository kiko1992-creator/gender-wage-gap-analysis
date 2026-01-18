-- Database initialization script for Gender Wage Gap Analysis
-- This runs automatically when PostgreSQL container first starts

-- Create countries table
CREATE TABLE IF NOT EXISTS countries (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    population BIGINT,
    gdp_billions NUMERIC(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create wage gap practice table
CREATE TABLE IF NOT EXISTS wage_gap_practice (
    id SERIAL PRIMARY KEY,
    country VARCHAR(100) NOT NULL,
    year INTEGER NOT NULL,
    gap_percent NUMERIC(5, 2),
    unemployment NUMERIC(5, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(country, year)
);

-- Create EU27 countries reference table
CREATE TABLE IF NOT EXISTS eu27_countries (
    id SERIAL PRIMARY KEY,
    country_code VARCHAR(2) UNIQUE NOT NULL,
    country_name VARCHAR(100) UNIQUE NOT NULL,
    join_year INTEGER,
    is_eurozone BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create comprehensive wage gap data table
CREATE TABLE IF NOT EXISTS wage_gap_data (
    id SERIAL PRIMARY KEY,
    country VARCHAR(100) NOT NULL,
    country_code VARCHAR(2),
    year INTEGER NOT NULL,
    wage_gap_percent NUMERIC(5, 2),
    male_hourly_earnings NUMERIC(10, 2),
    female_hourly_earnings NUMERIC(10, 2),
    gdp_per_capita NUMERIC(12, 2),
    unemployment_rate NUMERIC(5, 2),
    female_employment_rate NUMERIC(5, 2),
    tertiary_education_female NUMERIC(5, 2),
    data_source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(country, year)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_wage_gap_country ON wage_gap_practice(country);
CREATE INDEX IF NOT EXISTS idx_wage_gap_year ON wage_gap_practice(year);
CREATE INDEX IF NOT EXISTS idx_wage_data_country_year ON wage_gap_data(country, year);
CREATE INDEX IF NOT EXISTS idx_countries_name ON countries(name);

-- Create view for recent wage gap data
CREATE OR REPLACE VIEW recent_wage_gaps AS
SELECT
    country,
    year,
    gap_percent,
    unemployment,
    CASE
        WHEN gap_percent > 15 THEN 'High'
        WHEN gap_percent > 10 THEN 'Medium'
        ELSE 'Low'
    END as gap_category
FROM wage_gap_practice
WHERE year >= 2020
ORDER BY year DESC, gap_percent DESC;

-- Grant permissions (if needed for specific user)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Database tables created successfully!';
    RAISE NOTICE 'Tables: countries, wage_gap_practice, eu27_countries, wage_gap_data';
    RAISE NOTICE 'View: recent_wage_gaps';
END $$;
