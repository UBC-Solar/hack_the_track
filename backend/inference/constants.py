STATE_COLS   = ['accx', 'accy', 'speed', 'nmot', 'y', 'x']
CONTROL_COLS = ['gear', 'aps', 'pbrake_f', 'pbrake_r']
TELEMETRY_NAMES = [
    "accx_can", "accy_can", "speed", "gear", "aps", "nmot",
    "pbrake_f", "pbrake_r", "VBOX_Lat_Min", "VBOX_Long_Minutes",
]
EARTH_RADIUS = 6371000.0  # meters
SEQ_LEN = 10
SCALE = 50.0
