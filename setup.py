from setuptools import setup, find_packages

setup(
    name="fitness-analyzer",
    version="1.0.0",
    description="A comprehensive fitness analysis system using computer vision and AI",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        # Flask and extensions
        "Flask>=2.3.3",
        "Flask-Login>=0.6.2",
        "Flask-SQLAlchemy>=3.0.5",
        "Flask-WTF>=1.1.1",
        "Werkzeug>=2.3.7",
        "email-validator>=2.0.0",
        
        # Image processing and computer vision
        "opencv-python>=4.8.0",
        "numpy>=1.24.3",
        "Pillow>=10.0.0",
        "mediapipe>=0.10.3",
        
        # Deep learning and ML
        "tensorflow>=2.13.0",
        "tensorflow-hub>=0.14.0",
        "keras>=2.13.1",
        
        # Data handling and utilities
        "pandas>=2.0.3",
        "scipy>=1.11.2",
        "scikit-learn>=1.3.0",
        
        # Image enhancement and processing
        "albumentations>=1.3.1",
        "imgaug>=0.4.0",
        
        # Visualization
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ],
        "prod": [
            "gunicorn>=21.2.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "fitness-analyzer=app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 