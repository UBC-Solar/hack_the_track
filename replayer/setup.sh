cd "$(dirname "$0")"

curl -o R1_barber_telemetry_data.csv https://drive.google.com/file/d/1tYBQDdOsisdJZjR1xPHmp3jN5iDgz2HR/view?usp=share_link
mv R1_barber_telemetry_data.csv raw_data
uv run process_csv.py