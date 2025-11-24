cd "$(dirname "$0")"

FILE_ID="1tYBQDdOsisdJZjR1xPHmp3jN5iDgz2HR"
FILE_NAME="R1_barber_telemetry_data.csv"

curl -c ./cookie -s -L \
  "https://drive.usercontent.google.com/download?id=${FILE_ID}&confirm=t" \
  -o /dev/null

CONFIRM=$(awk '/download/ {print $NF}' cookie)

curl -Lb ./cookie \
  "https://drive.usercontent.google.com/download?id=${FILE_ID}&confirm=${CONFIRM}" \
  -o "${FILE_NAME}"

mkdir raw_data

mv R1_barber_telemetry_data.csv raw_data/

uv run process_csv.py