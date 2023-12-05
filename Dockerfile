FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r ./requirements.txt

RUN pip3 install pip
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY src /app/src/

CMD ["gunicorn", "src.core.main:app", "--bind", "0.0.0.0:8080", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]
