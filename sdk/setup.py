from setuptools import setup, find_packages

setup(
    name="modelforge-sdk",
    version="1.0.0",
    description="Python Client SDK for ModelForge AI Enterprise Platform",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "httpx>=0.25.0",
        "numpy>=1.24.0"
    ]
)
