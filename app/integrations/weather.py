import logging
from datetime import date
import requests

logger = logging.getLogger(__name__)

OPEN_METEO_BASE = 'https://api.open-meteo.com/v1/forecast'

DEFAULT_LOCATIONS = [
    {'name': 'San Diego', 'lat': 32.7157, 'lon': -117.1611},
    {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
    {'name': 'New Delhi', 'lat': 28.6139, 'lon': 77.2090},
]

# WMO Weather interpretation codes
WMO_CODES = {
    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
    45: 'Foggy', 48: 'Rime fog', 51: 'Light drizzle', 53: 'Moderate drizzle',
    55: 'Dense drizzle', 61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
    71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow', 80: 'Slight showers',
    81: 'Moderate showers', 82: 'Violent showers', 95: 'Thunderstorm',
    96: 'Thunderstorm w/ hail', 99: 'Thunderstorm w/ heavy hail',
}


class WeatherService:
    def fetch_weather(self, locations=None):
        """Fetch daily weather for each location. Returns list of dicts."""
        locations = locations or DEFAULT_LOCATIONS
        results = []

        for loc in locations:
            try:
                params = {
                    'latitude': loc['lat'],
                    'longitude': loc['lon'],
                    'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode',
                    'timezone': 'auto',
                    'forecast_days': 3,
                    'temperature_unit': 'fahrenheit',
                }
                resp = requests.get(OPEN_METEO_BASE, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                results.append({
                    'location_name': loc['name'],
                    'latitude': loc['lat'],
                    'longitude': loc['lon'],
                    'date': date.today(),
                    'data_json': data,
                })
            except Exception as e:
                logger.warning(f"Weather fetch failed for {loc['name']}: {e}")

        return results

    def format_weather_section(self, weather_entries):
        """Format cached weather data into readable brief content."""
        items = []
        for entry in weather_entries:
            data = entry.get('data_json', {})
            daily = data.get('daily', {})
            if not daily:
                continue

            temps_max = daily.get('temperature_2m_max', [])
            temps_min = daily.get('temperature_2m_min', [])
            codes = daily.get('weathercode', [])
            dates = daily.get('time', [])

            forecasts = []
            for i in range(min(3, len(dates))):
                condition = WMO_CODES.get(codes[i] if i < len(codes) else 0, 'Unknown')
                high = temps_max[i] if i < len(temps_max) else '?'
                low = temps_min[i] if i < len(temps_min) else '?'
                forecasts.append({
                    'date': dates[i] if i < len(dates) else '',
                    'condition': condition,
                    'high_f': high,
                    'low_f': low,
                })

            items.append({
                'location': entry.get('location_name', ''),
                'forecasts': forecasts,
            })

        return items
