FROM python:3.8.12-buster
WORKDIR /prod
COPY wineteller wineteller
COPY raw_data raw_data
#### Add raw data : csv files, trained models
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install .
COPY .env .env
CMD uvicorn wineteller.api.fast:app --host 0.0.0.0 --port $PORT
