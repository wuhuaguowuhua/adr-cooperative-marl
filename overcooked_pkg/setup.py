from setuptools import setup, find_packages
setup(
    name="overcooked_seac",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["gym==0.21.0", "overcooked-ai"],
)
