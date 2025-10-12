# Downloading UCF-Crime

1. Request access to the dataset from the official site: http://crcv.ucf.edu/data/UCF-Crime/
2. Once approved, download the RGB videos and metadata archives. Recommended layout:
   - `Training_Normal_Videos.zip`
   - `Training_Anomaly_Videos.zip`
   - `Testing_Normal_Videos.zip`
   - `Testing_Anomaly_Videos.zip`
   - `Anomaly_Train.txt`
   - `Anomaly_Test.txt`
3. Extract all archives into `data/ucf_crime/` so that the structure matches the README in that folder.
4. (Optional) If you have optical flow data, place it under `data/ucf_crime/flows/` and update the loaders accordingly.
5. Verify checksums using the MD5 hashes provided by the dataset maintainers.
6. Update `configs/thresholds.yaml` if you calibrate new operating points from your validation split.
