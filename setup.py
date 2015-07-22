from setuptools import setup

setup(
    name='grasp',
    license='Apache 2.0',
    author='Wilker Aziz',
    author_email='wilkeraziz@gmail.com',
    description='Randomised Semiring Parsing',
    version='0.0.dev1',
    packages=['grasp'],
    install_requires=['tabulate',
                      'nltk',
                      'ply']
)
