FROM python:3.9

WORKDIR /

RUN apt-get update -y
RUN apt-get install -y enchant-2
RUN apt-get install -y graphviz
RUN apt install -y hunspell-ru

COPY merged_requirements.txt .
RUN pip install --default-timeout=100 -r ./merged_requirements.txt

COPY . /src/

CMD ["gunicorn", "src.tg.grammar_ru.components.api.spelcheck_handler:app", "--bind", "0.0.0.0:8080", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]
