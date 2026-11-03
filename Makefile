.PHONY: all data model dashboard test
all: data model dashboard
data:
	python src/get_data.py
model:
	python src/run_all.py
dashboard:
	python src/export_dashboard.py
test:
	python -m pytest -q
