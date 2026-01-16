"""
Sample EU27 wage gap data for fallback when database is unavailable
Real Eurostat data (2020-2023)
"""

import pandas as pd

def get_sample_countries_2023():
    """Return sample EU27 countries data for 2023"""
    data = {
        'country_name': [
            'Latvia', 'Estonia', 'Slovakia', 'Austria', 'Germany', 'Hungary',
            'Finland', 'Czechia', 'France', 'Netherlands', 'Denmark', 'Portugal',
            'Lithuania', 'Cyprus', 'Spain', 'Sweden', 'Greece', 'Ireland',
            'Bulgaria', 'Malta', 'Croatia', 'Slovenia', 'Belgium', 'Italy',
            'Poland', 'Romania', 'Luxembourg'
        ],
        'region': [
            'Northern Europe', 'Northern Europe', 'Eastern Europe', 'Western Europe',
            'Western Europe', 'Eastern Europe', 'Northern Europe', 'Eastern Europe',
            'Western Europe', 'Western Europe', 'Northern Europe', 'Southern Europe',
            'Northern Europe', 'Southern Europe', 'Southern Europe', 'Northern Europe',
            'Southern Europe', 'Western Europe', 'Eastern Europe', 'Southern Europe',
            'Southern Europe', 'Southern Europe', 'Western Europe', 'Southern Europe',
            'Eastern Europe', 'Eastern Europe', 'Western Europe'
        ],
        'population': [
            1884000, 1331000, 5460000, 9042000, 83200000, 9689000,
            5536000, 10700000, 67750000, 17530000, 5857000, 10330000,
            2795000, 1244000, 47420000, 10490000, 10640000, 5033000,
            6877000, 525000, 3900000, 2108000, 11590000, 59030000,
            37750000, 19050000, 640000
        ],
        'gdp_billions': [
            40.0, 38.0, 115.0, 477.0, 4260.0, 177.0,
            297.0, 281.0, 2938.0, 1013.0, 395.0, 254.0,
            67.0, 28.0, 1426.0, 585.0, 215.0, 529.0,
            84.0, 17.0, 68.0, 62.0, 594.0, 2101.0,
            688.0, 301.0, 85.0
        ],
        'wage_gap_percent': [
            21.2, 20.5, 18.9, 18.8, 17.6, 17.3,
            16.5, 16.4, 15.0, 13.8, 13.0, 13.2,
            13.3, 11.5, 11.7, 11.8, 12.0, 11.3,
            12.4, 11.1, 10.8, 8.7, 5.8, 5.0,
            4.5, 3.6, 0.7
        ]
    }
    return pd.DataFrame(data)

def get_sample_country_trend(country_name):
    """Return 4-year trend for a specific country"""
    trends = {
        'Latvia': [(2020, 22.9), (2021, 22.3), (2022, 21.6), (2023, 21.2)],
        'Estonia': [(2020, 22.3), (2021, 21.8), (2022, 21.1), (2023, 20.5)],
        'Slovakia': [(2020, 20.2), (2021, 19.8), (2022, 19.4), (2023, 18.9)],
        'Austria': [(2020, 19.0), (2021, 18.9), (2022, 18.8), (2023, 18.8)],
        'Germany': [(2020, 18.6), (2021, 18.3), (2022, 18.0), (2023, 17.6)],
        'Hungary': [(2020, 18.8), (2021, 18.2), (2022, 17.8), (2023, 17.3)],
        'Finland': [(2020, 17.7), (2021, 17.2), (2022, 16.9), (2023, 16.5)],
        'Czechia': [(2020, 17.5), (2021, 17.0), (2022, 16.9), (2023, 16.4)],
        'France': [(2020, 16.0), (2021, 15.5), (2022, 15.2), (2023, 15.0)],
        'Netherlands': [(2020, 14.6), (2021, 14.3), (2022, 14.0), (2023, 13.8)],
        'Denmark': [(2020, 14.0), (2021, 13.5), (2022, 13.2), (2023, 13.0)],
        'Portugal': [(2020, 13.3), (2021, 12.9), (2022, 13.0), (2023, 13.2)],
        'Lithuania': [(2020, 14.8), (2021, 14.4), (2022, 13.9), (2023, 13.3)],
        'Cyprus': [(2020, 12.4), (2021, 12.1), (2022, 11.8), (2023, 11.5)],
        'Spain': [(2020, 12.0), (2021, 12.2), (2022, 11.9), (2023, 11.7)],
        'Sweden': [(2020, 12.3), (2021, 12.2), (2022, 12.1), (2023, 11.8)],
        'Greece': [(2020, 12.4), (2021, 12.5), (2022, 12.6), (2023, 12.0)],
        'Ireland': [(2020, 12.2), (2021, 11.9), (2022, 11.5), (2023, 11.3)],
        'Bulgaria': [(2020, 14.0), (2021, 14.1), (2022, 13.0), (2023, 12.4)],
        'Malta': [(2020, 12.2), (2021, 11.8), (2022, 11.5), (2023, 11.1)],
        'Croatia': [(2020, 11.8), (2021, 11.5), (2022, 11.2), (2023, 10.8)],
        'Slovenia': [(2020, 9.5), (2021, 9.3), (2022, 9.1), (2023, 8.7)],
        'Belgium': [(2020, 5.0), (2021, 5.3), (2022, 5.6), (2023, 5.8)],
        'Italy': [(2020, 4.7), (2021, 5.0), (2022, 5.2), (2023, 5.0)],
        'Poland': [(2020, 8.5), (2021, 5.4), (2022, 4.8), (2023, 4.5)],
        'Romania': [(2020, 3.5), (2021, 3.6), (2022, 3.8), (2023, 3.6)],
        'Luxembourg': [(2020, 1.4), (2021, 1.3), (2022, 1.0), (2023, 0.7)]
    }

    if country_name in trends:
        return pd.DataFrame(trends[country_name], columns=['year', 'wage_gap_percent'])
    return pd.DataFrame(columns=['year', 'wage_gap_percent'])

def get_sample_regional_comparison():
    """Return regional averages for 2023"""
    data = {
        'region': ['Northern Europe', 'Eastern Europe', 'Western Europe', 'Southern Europe'],
        'num_countries': [6, 7, 7, 7],
        'avg_gap': [16.05, 12.18, 11.86, 10.50],
        'min_gap': [11.8, 3.6, 0.7, 5.0],
        'max_gap': [21.2, 18.9, 18.8, 13.2]
    }
    return pd.DataFrame(data)

def get_sample_improvement_rankings():
    """Return countries ranked by improvement (2020 → 2023)"""
    data = {
        'country_name': [
            'Poland', 'Estonia', 'Latvia', 'Bulgaria', 'Lithuania',
            'Slovakia', 'Hungary', 'Finland', 'Germany', 'Czechia',
            'France', 'Denmark', 'Netherlands', 'Croatia', 'Ireland',
            'Cyprus', 'Slovenia', 'Sweden', 'Austria', 'Portugal',
            'Romania', 'Italy', 'Belgium', 'Greece', 'Malta', 'Spain', 'Luxembourg'
        ],
        'gap_2020': [
            8.5, 22.3, 22.9, 14.0, 14.8, 20.2, 18.8, 17.7, 18.6, 17.5,
            16.0, 14.0, 14.6, 11.8, 12.2, 12.4, 9.5, 12.3, 19.0, 13.3,
            3.5, 4.7, 5.0, 12.4, 12.2, 12.0, 1.4
        ],
        'gap_2023': [
            4.5, 20.5, 21.2, 12.4, 13.3, 18.9, 17.3, 16.5, 17.6, 16.4,
            15.0, 13.0, 13.8, 10.8, 11.3, 11.5, 8.7, 11.8, 18.8, 13.2,
            3.6, 5.0, 5.8, 12.0, 11.1, 11.7, 0.7
        ],
        'change': [
            -4.0, -1.8, -1.7, -1.6, -1.5, -1.3, -1.5, -1.2, -1.0, -1.1,
            -1.0, -1.0, -0.8, -1.0, -0.9, -0.9, -0.8, -0.5, -0.2, -0.1,
            0.1, 0.3, 0.8, -0.4, -1.1, -0.3, -0.7
        ]
    }
    return pd.DataFrame(data).sort_values('change')
