from telemetry.raw.TelemetryDB import TelemetryDB
from matplotlib import pyplot as plt

db = TelemetryDB("postgresql+psycopg2://racer:changeme@100.120.36.75:5432/racing")


#Available telemetry signals: ['accx_can', 'accy_can', 'ath', 'gear', 'nmot', 'pbrake_f', 'pbrake_r', 'speed', 'Steering_Angle'

# telemetry data for car 2 - GR86-002-000
car = db.get_car_race(track="barber", race_number=2, vehicle_code="GR86-002-000")
gps = db.get_gps_race(track="barber", race_number=2, vehicle_code="GR86-002-000")

if car:
    df_accx = car.get_telemetry("accx_can")
    df_accy = car.get_telemetry("accy_can")
    df_speed = car.get_telemetry("speed")
    df_ath = car.get_telemetry("ath_can")
    df_gear = car.get_telemetry("gear")
    df_nmotor = car.get_telemetry("nmot")
    df_pbrake_f = car.get_telemetry("pbrake_f")
    df_pbrake_r = car.get_telemetry("pbrake_r")
    df_steering = car.get_telemetry("Steering_Angle")


#car 1 might be faulty, did not really finish lap2

df_ath.head()





#consider state and control inputs




