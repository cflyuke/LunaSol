from setuptools import setup, find_packages

setup(
    name="LunaSol",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "spiceypy",
        "pandas",
        "matplotlib",
        "cartopy",
        "opencv-python",
        "scipy",
        "tqdm"
    ],
    author="cflyuke",
    author_email="cflyuke@gmail.com",
    description="A Python-based astronomical model for predicting and analyzing solar and lunar eclipses",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cflyuke/LunaSol.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    package_data={
        "LunaSol": ["data/*"]
    }
)
