# Exporting TensorFlow Lite Models

This project keeps the training graph in TensorFlow format. To run on edge devices, convert the exported SavedModel to TFLite.

1. Train and export models using `python -m surveillance_tf.export.save_models` (or `make export`).
2. Use the conversion snippet below to create a float16 TFLite model:

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("outputs/saved_models/movinet_mil")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open("outputs/tflite/movinet_mil_fp16.tflite", "wb") as f:
    f.write(tflite_model)
```

3. For EfficientDet-Lite overlays, start from the pre-converted models on TensorFlow Hub and store them under `outputs/detectors/`.
4. Validate latency with the Streamlit demo using sample clips.
