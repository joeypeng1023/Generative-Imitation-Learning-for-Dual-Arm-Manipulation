from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements-minimal.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="generative-imitation-dual-arm",
    version="1.0.0",
    author="SDSC6019 Group Project",
    author_email="",
    description="Generative Imitation Learning for Dual-Arm Manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/generative-imitation-dual-arm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "full": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "seaborn>=0.12.0",
            "plotly>=5.17.0",
            "moviepy>=1.0.3",
            "reportlab>=4.0.0",
            "pytest>=7.4.0",
            "black>=23.9.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gil-train-bc=training.bc_train:main",
            "gil-train-dagger=training.dagger_train:main",
            "gil-eval-bc=evaluation.eval_bc:main",
            "gil-eval-dagger=evaluation.eval_dagger:main",
            "gil-collect-data=data.collect_demonstrations_auto:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.json"],
    },
)