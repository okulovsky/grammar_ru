FROM python:3.11

RUN pip install annotated-types==0.6.0 anyio==4.3.0 certifi==2024.2.2 charset-normalizer==3.3.2 click==8.1.7 colorama==0.4.6 docker==7.0.0 fastapi==0.110.1 filelock==3.13.4 fsspec==2024.3.1 h11==0.14.0 httpcore==1.0.5 httptools==0.6.1 httpx==0.27.0 huggingface-hub==0.22.2 idna==3.6 iniconfig==2.0.0 numpy==1.26.4 nvidia-cublas-cu11==11.10.3.66 nvidia-cuda-nvrtc-cu11==11.7.99 nvidia-cuda-runtime-cu11==11.7.99 nvidia-cudnn-cu11==8.5.0.96 packaging==24.0 pluggy==1.4.0 pydantic==2.6.4 pydantic_core==2.16.3 pytest==8.1.1 python-dotenv==1.0.1 PyYAML==6.0.1 regex==2024.4.16 requests==2.31.0 safetensors==0.4.3 sniffio==1.3.1 starlette==0.37.2 tokenizers==0.19.1 torch==1.13.1 tqdm==4.66.2 transformers==4.40.1 typing_extensions==4.10.0 urllib3==2.2.1 uvicorn==0.29.0 watchfiles==0.21.0 websockets==12.0

COPY /src/ /workdir/src/

WORKDIR /workdir


CMD ["uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "80"]