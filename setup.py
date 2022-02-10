from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="vflow",
    version="0.1.1",
    author="Yu Group",
    author_email="chandan_singh@berkeley.edu",
    description="A framework for doing stability analysis with PCS.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yu-Group/pcs-pipeline",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'networkx',
        'pandas',
        'joblib',
        'pytest',
        'ray',
        'mlflow',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pylint==2.12.2',
            'tqdm',
            'scikit-learn >=0.23.0',  # 0.23+ only works on py3.6+)
        ],
        'notebooks': [
            'tqdm',
            'jupyter',
            'scikit-learn >=0.23.0',  # 0.23+ only works on py3.6+)
            'torch >= 1.0.0',
            'torchvision',
        ],
    },
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
