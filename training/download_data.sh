#!/usr/bin/env bash
cd "$(dirname "$0")"
mkdir -p data
cd data

curl -o barber-motorsports-park.zip https://trddev.com/hackathon-2025/barber-motorsports-park.zip
curl -o circuit-of-the-americas.zip https://trddev.com/hackathon-2025/circuit-of-the-americas.zip
curl -o indianapolis.zip https://trddev.com/hackathon-2025/indianapolis.zip
curl -o road-america.zip https://trddev.com/hackathon-2025/road-america.zip
curl -o sonoma.zip https://trddev.com/hackathon-2025/sonoma.zip
curl -o virginia-international-raceway.zip https://trddev.com/hackathon-2025/virginia-international-raceway.zip

sudo apt install parallel
ls *.zip | parallel unzip

rm *.zip
rm -r __MACOSX