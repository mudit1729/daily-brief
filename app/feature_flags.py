import os

_FLAGS = {}


def init_flags():
    for key, val in os.environ.items():
        if key.startswith('FF_'):
            flag_name = key[3:].lower()
            _FLAGS[flag_name] = val.lower() in ('true', '1', 'yes')


def is_enabled(flag_name: str) -> bool:
    return _FLAGS.get(flag_name, False)


def all_flags() -> dict:
    return dict(_FLAGS)


def set_flag(flag_name: str, value: bool):
    _FLAGS[flag_name] = value
