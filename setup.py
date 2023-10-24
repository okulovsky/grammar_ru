from setuptools import setup, find_packages

setup(name='grammar_ru_dev',
      version='0.0.0',
      description='',
      packages=find_packages(),
      install_requires= [
            'anyio==3.6.2',
             'argon2-cffi==21.3.0',
             'argon2-cffi-bindings==21.2.0',
             'arrow==1.2.3',
             'asttokens==2.2.1',
             'async-generator==1.10',
             'attrs==22.2.0',
             'backcall==0.2.0',
             'beautifulsoup4==4.11.1',
             'bleach==5.0.1',
             'boto3==1.26.54',
             'botocore==1.29.54',
             'browser-cookie3==0.17.1',
             'cffi==1.15.1',
             'charset-normalizer==2.1.1',
             'click==8.1.3',
             'comm==0.1.2',
             'contextlib2==21.6.0',
             'contourpy==1.0.7',
             'corus==0.9.0',
             'coverage==7.0.5',
             'cramjam==2.6.2',
             'cycler==0.11.0',
             'DAWG-Python==0.7.2',
             'debugpy==1.6.5',
             'decorator==5.1.1',
             'defusedxml==0.7.1',
             'Deprecated==1.2.13',
             'dill==0.3.6',
             'docopt==0.6.2',
             'entrypoints==0.4',
             'exceptiongroup==1.1.1',
             'executing==1.2.0',
             'fastjsonschema==2.16.2',
             'fastparquet==2023.1.0',
             'filelock==3.8.2',
             'Flask==2.1.0',
             'fonttools==4.38.0',
             'fqdn==1.5.1',
             'fsspec==2023.1.0',
             'google-pasta==0.2.0',
             'h11==0.14.0',
             'homoglyphs==2.0.4',
             'huggingface-hub==0.11.1',
             'idna==3.4',
             'importlib-metadata==4.13.0',
             'importlib-resources==5.10.2',
             'install==1.3.5',
             'ipykernel==6.20.2',
             'ipython==8.8.0',
             'ipython-genutils==0.2.0',
             'ipywidgets==8.0.4',
             'isoduration==20.11.0',
             'itsdangerous==2.1.2',
             'jedi==0.18.2',
             'jeepney==0.8.0',
             'Jinja2==3.1.2',
             'jmespath==1.0.1',
             'joblib==1.2.0',
             'jsonpickle==3.0.1',
             'jsonpointer==2.3',
             'jsonschema==4.17.3',
             'jupyter==1.0.0',
             'jupyter-console==6.4.4',
             'jupyter-events==0.6.3',
             'jupyter_client==7.4.9',
             'jupyter_core==5.1.3',
             'jupyter_server==2.1.0',
             'jupyter_server_terminals==0.4.4',
             'jupyterlab-pygments==0.2.2',
             'jupyterlab-widgets==3.0.5',
             'kiwisolver==1.4.4',
             'lxml==4.9.1',
             'lz4==4.3.2',
             'MarkupSafe==2.1.2',
             'matplotlib==3.6.3',
             'matplotlib-inline==0.1.6',
             'mistune==2.0.4',
             'multiprocess==0.70.14',
             'navec==0.10.0',
             'nbclassic==0.4.8',
             'nbclient==0.7.2',
             'nbconvert==7.2.8',
             'nbformat==5.7.3',
             'nerus==1.7.0',
             'nest-asyncio==1.5.6',
             'nltk==3.8.1',
             'notebook==6.5.2',
             'notebook_shim==0.2.2',
             'numpy==1.24.1',
             'nvidia-cublas-cu11==11.10.3.66',
             'nvidia-cuda-nvrtc-cu11==11.7.99',
             'nvidia-cuda-runtime-cu11==11.7.99',
             'nvidia-cudnn-cu11==8.5.0.96',
             'outcome==1.2.0',
             'packaging==23.0',
             'pandas==1.5.3',
             'pandocfilters==1.5.0',
             'parso==0.8.3',
             'pathos==0.3.0',
             'patsy==0.5.3',
             'pexpect==4.8.0',
             'pickleshare==0.7.5',
             'Pillow==9.4.0',
             'pkgutil_resolve_name==1.3.10',
             'platformdirs==2.6.2',
             'pox==0.3.2',
             'ppft==1.7.6.6',
             'prometheus-client==0.15.0',
             'prompt-toolkit==3.0.36',
             'protobuf==3.20.3',
             'protobuf3-to-dict==0.1.5',
             'psutil==5.9.4',
             'ptyprocess==0.7.0',
             'pure-eval==0.2.2',
             'pyaml==21.10.1',
             'pyarrow==10.0.1',
             'pycparser==2.21',
             'pycryptodomex==3.17',
             'pyenchant==3.2.2',
             'Pygments==2.14.0',
             'pymorphy2==0.9.1',
             'pymorphy2-dicts-ru==2.4.417127.4579844',
             'pyparsing==3.0.9',
             'pyrsistent==0.19.3',
             'PySocks==1.7.1',
             'python-dateutil==2.8.2',
             'python-dotenv==0.21.1',
             'python-json-logger==2.0.4',
             'pytz==2022.7.1',
             'PyYAML==6.0',
             'pyzmq==25.0.0',
             'qtconsole==5.4.0',
             'QtPy==2.3.0',
             'razdel==0.5.0',
             'regex==2022.10.31',
             'requests==2.28.1',
             'rfc3339-validator==0.1.4',
             'rfc3986-validator==0.1.1',
             's3transfer==0.6.0',
             'sagemaker==2.129.0',
             'schema==0.7.5',
             'scikit-learn==1.2.0',
             'scipy==1.10.0',
             'seaborn==0.12.2',
             'selenium==4.8.2',
             'Send2Trash==1.8.0',
             'sentencepiece==0.1.97',
             'simplejson==3.18.1',
             'six==1.16.0',
             'slovnet==0.5.0',
             'smdebug-rulesconfig==1.0.1',
             'sniffio==1.3.0',
             'snowballstemmer==2.2.0',
             'sortedcontainers==2.4.0',
             'soupsieve==2.3.2.post1',
             'stack-data==0.6.2',
             'statsmodels==0.13.5',
             'terminado==0.17.1',
             'threadpoolctl==3.1.0',
             'tinycss2==1.2.1',
             'tokenizers==0.12.1',
             'torch==1.13.1',
             'tornado==6.2',
             'tqdm==4.64.1',
             'traitlets==5.8.1',
             'transformers==4.22.2',
             'trio==0.22.0',
             'trio-websocket==0.10.2',
             'typing_extensions==4.4.0',
             'uri-template==1.2.0',
             'urllib3==1.26.14',
             'wcwidth==0.2.6',
             'webcolors==1.12',
             'webencodings==0.5.1',
             'websocket-client==1.4.2',
             'Werkzeug==2.2.2',
             'widgetsnbextension==4.0.5',
             'wrapt==1.14.1',
             'wsproto==1.2.0',
             'yo-fluq==1.1.14',
             'yo-fluq-ds==1.1.14',
             'zipp==3.11.0'
      ],
      include_package_data = True,
      zip_safe=False
      )
