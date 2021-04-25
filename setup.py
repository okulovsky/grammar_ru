from setuptools import setup, find_packages
setup(name='grammar_ru',
      version='0.0.0',
      description='Demo',
      packages=find_packages(),
      install_requires=[
        'training_grounds',
        'slovnet',
      ],
      include_package_data = True,
      zip_safe=False
)