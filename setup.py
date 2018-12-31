try:
    import mycodecpy
except:
    import warnings
    warnings.warn('Cython_IPEM not found, '
                  'you will only be able to analyze data files.')

from setuptools import setup

setup(
    name='LEGION-Nengo',
    version='1.0.0',
    description='Terman-Wang neural oscillator & LEGION',
    url='https://github.com/theandychung/LEGION-Nengo',
    install_requires=[
        'nengo',
        'matplotlib',
        'pandas',
        'numpy'
    ]
)



