from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from app.extensions import db
from app.models.system_setting import SystemSetting


DEFAULT_PIPELINE_SCHEDULE = {
    'enabled': True,
    'hour': 5,
    'minute': 30,
    'timezone': 'UTC',
}


class SchedulerConfigService:
    PIPELINE_KEY = 'pipeline_schedule'

    def get_pipeline_schedule(self):
        raw = SystemSetting.get_value(self.PIPELINE_KEY, DEFAULT_PIPELINE_SCHEDULE)
        return self._normalize_pipeline_schedule(raw, strict=False)

    def update_pipeline_schedule(self, updates):
        current = self.get_pipeline_schedule()
        merged = {**current, **updates}
        normalized = self._normalize_pipeline_schedule(merged, strict=True)
        SystemSetting.set_value(self.PIPELINE_KEY, normalized)
        db.session.commit()
        return normalized

    def _normalize_pipeline_schedule(self, payload, strict=False):
        data = payload if isinstance(payload, dict) else {}
        normalized = dict(DEFAULT_PIPELINE_SCHEDULE)

        enabled = data.get('enabled', normalized['enabled'])
        if isinstance(enabled, bool):
            normalized['enabled'] = enabled
        elif strict:
            raise ValueError('"enabled" must be a boolean')

        for field, max_value in (('hour', 23), ('minute', 59)):
            raw = data.get(field, normalized[field])
            try:
                value = int(raw)
            except (TypeError, ValueError):
                if strict:
                    raise ValueError(f'"{field}" must be an integer')
                continue

            if value < 0 or value > max_value:
                if strict:
                    raise ValueError(f'"{field}" must be between 0 and {max_value}')
                continue

            normalized[field] = value

        timezone_name = str(data.get('timezone', normalized['timezone']) or '').strip()
        if timezone_name:
            try:
                ZoneInfo(timezone_name)
                normalized['timezone'] = timezone_name
            except ZoneInfoNotFoundError:
                if strict:
                    raise ValueError('Invalid timezone')
        elif strict:
            raise ValueError('"timezone" is required')

        return normalized

