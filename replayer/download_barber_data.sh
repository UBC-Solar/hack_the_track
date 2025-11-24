cd "$(dirname "$0")"

FILE_ID="1NDJD1yRljBcoH_IwsjH9WiYhxyfd0QBx"
FILE_NAME="replay_ready_r1_barber.csv"

curl -c ./cookie -s -L \
  "https://drive.usercontent.google.com/download?id=${FILE_ID}&confirm=t" \
  -o /dev/null

CONFIRM=$(awk '/download/ {print $NF}' cookie)

curl -Lb ./cookie \
  "https://drive.usercontent.google.com/download?id=${FILE_ID}&confirm=${CONFIRM}" \
  -o "${FILE_NAME}"

mkdir replayer_data

mv ${FILE_NAME} replayer_data/