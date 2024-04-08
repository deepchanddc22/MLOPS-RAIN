import os

def create_directory_structure(root_dir):
    # Create the root directory
    os.makedirs(root_dir, exist_ok=True)

    # Create subdirectories
    subdirectories = [
        'deeplearning_package',
        'deeplearning_package/data',
        'deeplearning_package/models',
        'deeplearning_package/training',
        'deeplearning_package/utils'
    ]

    for subdir in subdirectories:
        os.makedirs(os.path.join(root_dir, subdir), exist_ok=True)

    # Create __init__.py files
    init_files = [
        'deeplearning_package/__init__.py',
        'deeplearning_package/data/__init__.py',
        'deeplearning_package/models/__init__.py',
        'deeplearning_package/training/__init__.py',
        'deeplearning_package/utils/__init__.py'
    ]

    for init_file in init_files:
        with open(os.path.join(root_dir, init_file), 'a'):
            pass  # Create empty __init__.py file

    # Create setup.py
    setup_py_content = """from setuptools import setup, find_packages

setup(
    name='deeplearning_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List dependencies here
        'numpy',
        'tensorflow',  # or 'torch' for PyTorch
        # Add other dependencies as needed
    ],
)
"""
    with open(os.path.join(root_dir, 'setup.py'), 'w') as f:
        f.write(setup_py_content)

    # Create README.md
    readme_content = "# Deep Learning Package\n\nWrite your project description here."
    with open(os.path.join(root_dir, 'README.md'), 'w') as f:
        f.write(readme_content)

if __name__ == "__main__":
    root_directory = input("Enter the path where you want to create the directory structure: ")
    create_directory_structure(root_directory)
    print(f"Directory structure created at: {root_directory}")
