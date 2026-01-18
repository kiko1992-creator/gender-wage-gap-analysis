-- Seed data for initial testing and development
-- Insert EU27 countries

INSERT INTO eu27_countries (country_code, country_name, join_year, is_eurozone) VALUES
('AT', 'Austria', 1995, true),
('BE', 'Belgium', 1958, true),
('BG', 'Bulgaria', 2007, false),
('HR', 'Croatia', 2013, true),
('CY', 'Cyprus', 2004, true),
('CZ', 'Czech Republic', 2004, false),
('DK', 'Denmark', 1973, false),
('EE', 'Estonia', 2004, true),
('FI', 'Finland', 1995, true),
('FR', 'France', 1958, true),
('DE', 'Germany', 1958, true),
('GR', 'Greece', 1981, true),
('HU', 'Hungary', 2004, false),
('IE', 'Ireland', 1973, true),
('IT', 'Italy', 1958, true),
('LV', 'Latvia', 2004, true),
('LT', 'Lithuania', 2004, true),
('LU', 'Luxembourg', 1958, true),
('MT', 'Malta', 2004, true),
('NL', 'Netherlands', 1958, true),
('PL', 'Poland', 2004, false),
('PT', 'Portugal', 1986, true),
('RO', 'Romania', 2007, false),
('SK', 'Slovakia', 2004, true),
('SI', 'Slovenia', 2004, true),
('ES', 'Spain', 1986, true),
('SE', 'Sweden', 1995, false)
ON CONFLICT (country_code) DO NOTHING;

-- Insert sample countries data
INSERT INTO countries (name, population, gdp_billions) VALUES
('Germany', 83200000, 4259.93),
('France', 67500000, 2937.47),
('Italy', 59100000, 2010.43),
('Spain', 47400000, 1461.55),
('Poland', 38000000, 688.18),
('Romania', 19200000, 301.34),
('Netherlands', 17500000, 1012.85),
('Belgium', 11600000, 594.10),
('Czech Republic', 10500000, 281.78),
('Greece', 10600000, 214.91),
('Portugal', 10300000, 251.94),
('Sweden', 10400000, 585.94),
('Hungary', 9700000, 181.85),
('Austria', 9000000, 477.03),
('Bulgaria', 6900000, 84.05),
('Denmark', 5900000, 395.10),
('Finland', 5500000, 297.30),
('Slovakia', 5500000, 115.48),
('Ireland', 5000000, 504.18),
('Croatia', 3900000, 68.66),
('Lithuania', 2800000, 65.50),
('Slovenia', 2100000, 63.64),
('Latvia', 1900000, 40.83),
('Estonia', 1300000, 38.10),
('Cyprus', 1200000, 28.41),
('Luxembourg', 640000, 85.50),
('Malta', 520000, 17.17)
ON CONFLICT (name) DO NOTHING;

-- Insert sample wage gap data
INSERT INTO wage_gap_practice (country, year, gap_percent, unemployment) VALUES
('Germany', 2023, 13.2, 3.1),
('Germany', 2022, 13.6, 3.0),
('Germany', 2021, 14.1, 3.6),
('France', 2023, 13.7, 7.3),
('France', 2022, 14.0, 7.3),
('Poland', 2023, 4.5, 2.9),
('Poland', 2022, 4.8, 2.9),
('Croatia', 2023, 10.2, 6.1),
('Croatia', 2022, 10.8, 7.0),
('Spain', 2023, 11.6, 12.2),
('Spain', 2022, 12.2, 12.9)
ON CONFLICT (country, year) DO NOTHING;

-- Log completion
DO $$
DECLARE
    country_count INTEGER;
    wage_gap_count INTEGER;
    eu27_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO country_count FROM countries;
    SELECT COUNT(*) INTO wage_gap_count FROM wage_gap_practice;
    SELECT COUNT(*) INTO eu27_count FROM eu27_countries;

    RAISE NOTICE 'Seed data inserted successfully!';
    RAISE NOTICE 'Countries: %', country_count;
    RAISE NOTICE 'EU27 Countries: %', eu27_count;
    RAISE NOTICE 'Wage Gap Records: %', wage_gap_count;
END $$;
