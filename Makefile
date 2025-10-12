PYTHON ?= python3
VENV ?=.venv
RUN = $(VENV)/bin/python

.PHONY: env train eval demo ui export tests

env:
	$(PYTHON) -m venv $(VENV)
	$(RUN) -m pip install --upgrade pip
	$(RUN) -m pip install -r requirements.txt
	$(RUN) -m pip install -e .

train:
	$(RUN) -m surveillance_tf.train.train_mil_ucfcrime \
		--data_root data/dcsass \
		--train_csv data/dcsass/splits/train.csv \
		--val_csv   data/dcsass/splits/val.csv \
		--out outputs/dcsass_run

eval:
	$(RUN) -m surveillance_tf.train.eval_mil_ucfcrime \
		--data_root data/dcsass \
		--test_csv data/dcsass/splits/test.csv \
		--ckpt models/movinet/ckpt_best \
		--out outputs/dcsass_run

demo:
	$(RUN) -m surveillance_tf.demo.coord \
		--data_root data/dcsass \
		--video "data/dcsass/sample/*.mp4" \
		--ckpt models/movinet/ckpt_best \
		--config configs/thresholds.yaml --fps 25

ui:
	STREAMLIT_BROWSER=0 $(RUN) -m streamlit run surveillance_tf/demo/ui_app.py --server.headless true

export:
	$(RUN) -m surveillance_tf.export.save_models

tests:
	$(RUN) -m pytest -q
