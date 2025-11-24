STATE_COLS   = ['accx', 'accy', 'speed', 'nmot', 'y', 'x']
CONTROL_COLS = ['gear', 'aps', 'pbrake_f', 'pbrake_r']
TELEMETRY_NAMES = [
    "accx_can", "accy_can", "speed", "gear", "aps", "nmot",
    "pbrake_f", "pbrake_r", "VBOX_Lat_Min", "VBOX_Long_Minutes",
]
EARTH_RADIUS = 6371000.0  # meters
SEQ_LEN = 10
SCALE = 50.0

R_EARTH = 6371000.0  # meters
telemetry_names = [
    "accx", "accy", "speed", "gear", "aps",
    "nmot", "pbrake_f", "pbrake_r", "latitude", "longitude",
]

DB_URL = "postgresql+psycopg2://racer:changeme@100.120.36.75:5432/racing"
state   = ["accx", "accy", "speed", "nmot", "latitude", "longitude"]
control = ["gear", "aps", "pbrake_f", "pbrake_r", "steering_angle"]
