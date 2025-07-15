from setuptools import setup, find_packages

setup(
    name='Sampling_SDK',
    version='0.1.0',
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.4',
        'pandas==2.3.0',
        'scikit-learn==1.3.2',
        'imbalanced-learn==0.11.0',
        'matplotlib',
        'seaborn'
    ],
    description='Sampling SDK for resampling, class imbalance handling, and data splitting',
)
