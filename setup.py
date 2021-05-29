from setuptools import setup, find_packages
setup(name='grammar_ru',
      version='0.0.0',
      description='Demo',
      packages=find_packages(),
      install_requires=[
          'slovnet==0.5.0',
          'pyenchant==3.2.0',
          'pymorphy2==0.9.1',
          'training_grounds',
          'nerus'
      ],
      include_package_data=True,
      zip_safe=False
      )
