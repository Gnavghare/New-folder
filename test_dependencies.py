"""
Test script to verify that all dependencies are installed correctly.
"""
import sys
import importlib

# List of packages to test
packages = [
    'flask',
    'opencv-python',
    'numpy',
    'tensorflow',
    'matplotlib',
    'pillow',
    'sklearn',
    'werkzeug',
    'pandas',
    'flask_wtf',
    'flask_sqlalchemy',
    'flask_migrate',
    'flask_login',
    'email_validator',
    'dotenv',
    'requests',
    'tqdm',
    'imutils',
    'tensorflow_hub'
]

# Check each package
print("Testing dependencies...")
print("-" * 50)

all_passed = True
for package in packages:
    try:
        # Handle special cases
        if package == 'opencv-python':
            module = importlib.import_module('cv2')
        elif package == 'pillow':
            module = importlib.import_module('PIL')
        elif package == 'dotenv':
            module = importlib.import_module('dotenv')
        else:
            module = importlib.import_module(package)
        
        # Get version if available
        version = getattr(module, '__version__', 'unknown version')
        print(f"✓ {package} ({version}) - Successfully imported")
    except ImportError as e:
        print(f"✗ {package} - Failed to import: {e}")
        all_passed = False

print("-" * 50)
if all_passed:
    print("All dependencies are installed correctly!")
else:
    print("Some dependencies are missing or not installed correctly.")
    
print("\nPython version:", sys.version) 