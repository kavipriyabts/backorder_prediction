from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    try:
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.strip() for req in requirements]  # Use strip() to remove whitespace and newlines

            if HYPEN_E_DOT in requirements:
                requirements.remove(HYPEN_E_DOT)

    except FileNotFoundError:
        print(f"Warning: {file_path} not found. No requirements will be installed.")
    
    return requirements

setup(
    name='Back_order_prediction_end2end',
    version='0.0.1',
    author='Kavipriya Sekar',
    author_email='kavipriyasekar@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages(),
    description='A package for predicting back orders in an end-to-end manner.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the Python version required
)