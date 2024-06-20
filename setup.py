from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="annime",
    version="0.1.0.1",
    author="Avgustin Zhugalov",
    author_email="avgustinalex@yandex.ru",
    description="ANN-interface library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/Azherhs/Annime",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    keywords="ann interface metrics",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pytest",
        "matplotlib",
        "h5py",
        "PyYAML",
        "hnswlib",
        "datasketch",
        "faiss-cpu",
        "ngt",
        "scann",
        "annoy",
        "nmslib"
    ],
)
