from telemetry.raw.TelemetryDB import TelemetryDB

db = TelemetryDB("postgresql+psycopg2://racer:changeme@100.120.36.75:5432/racing")

# # Show all races/vehicles in the database
# for car_race in db.list_car_races():
#     print(f"Race {car_race.race_number} at {car_race.track_name} → {car_race.vehicle_code}")
#
# # Pick one
car = db.get_car_race(track="cota", race_number=1, vehicle_code="GR86-002-2")
if car:
    print("Available telemetry signals:", car.list_telemetry_names())
    #df = car.get_telemetry("accx_can")
    df = car.get_telemetry_10s("accx_can")
    print(df.head())


#car 1 might be faulty, did not really finish lap2


#consider state and control inputs







