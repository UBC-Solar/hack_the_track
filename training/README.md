# Training

To train the recurrent neural network, you'll need to download the race data and upload it to a PostgreSQL database for it to be accessible.

```bash
./download_data.sh
docker compose up -d
uv run ingest_data.py
```

After that, the `train_rnn.py` notebook should be usable! Finished models should be put into `backend/inference/models` to be accessible.