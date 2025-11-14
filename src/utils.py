import yaml

def load_settings():
    return yaml.safe_load(open("config/settings.yaml"))
