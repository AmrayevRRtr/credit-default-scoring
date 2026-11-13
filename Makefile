install:
	pip install -r requirements.txt
train:
	python -m creditrisk.pipeline
verify:
	python verify_results.py
test:
	python -m pytest -q tests
app:
	streamlit run app/streamlit_app.py
