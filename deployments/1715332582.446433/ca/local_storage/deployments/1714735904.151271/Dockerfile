FROM python:3.8

RUN pip install torch torchvision

COPY . /

RUN pip install -e .

CMD ["python3","/entry.py"]
