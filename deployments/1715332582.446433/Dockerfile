FROM python:3.8

RUN pip install blas==1.0 brotli-python==1.0.9 bzip2==1.0.8 ca-certificates==2024.3.11 certifi==2024.2.2 cffi==1.16.0 charset-normalizer==2.0.4 filelock==3.13.1 freetype==2.12.1 fsspec==2024.3.1 gmpy2==2.1.2 idna==3.7 intel-openmp==2023.1.0 jinja2==3.1.3 jpeg==9e lcms2==2.12 lerc==3.0 libdeflate==1.17 libffi==3.4.4 libpng==1.6.39 libtiff==4.5.1 libuv==1.44.2 libwebp-base==1.3.2 lz4-c==1.9.4 markupsafe==2.1.3 mkl==2023.1.0 mkl-service==2.4.0 mkl_fft==1.3.8 mkl_random==1.2.4 mpc==1.1.0 mpfr==4.0.2 mpir==3.0.0 mpmath==1.3.0 networkx==3.1 ninja==1.10.2 ninja-base==1.10.2 numpy==1.26.4 numpy-base==1.26.4 openjpeg==2.4.0 openssl==3.0.13 pillow==10.3.0 pip==24.0 pycparser==2.21 pysocks==1.7.1 python==3.11.9 pytorch==2.2.0 requests==2.31.0 setuptools==69.5.1 sqlite==3.45.3 sympy==1.12 tbb==2021.8.0 tk==8.6.14 torchvision==0.15.2 typing_extensions==4.11.0 tzdata==2024a urllib3==2.1.0 vc==14.2 vs2015_runtime==14.27.29016 wheel==0.43.0 win_inet_pton==1.1.0 xz==5.4.6 zlib==1.2.13 zstd==1.5.5


COPY . /

RUN pip install -e .

CMD ["python3","/entry.py"]
