import setuptools

setuptools.setup(
    name="pcs",
    version="0.2.6",
    author="Yu Group",
    author_email="chandan_singh@berkeley.edu",
    description="A framework for doing stability analysis with PCS.",
    long_description="A framework for doing stability analysis with PCS.",
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/imodels",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'scikit-learn>=0.23.0', # 0.23+ only works on py3.6+)
        'pytest',
    ],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
