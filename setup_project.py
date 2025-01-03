import os

# Create directory structure
directories = [
    'src',
    'src/data',
    'src/models',
    'src/recommenders',
    'src/utils'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    
# Create empty __init__.py files
init_files = [
    'src/__init__.py',
    'src/data/__init__.py',
    'src/models/__init__.py',
    'src/recommenders/__init__.py',
    'src/utils/__init__.py'
]

for init_file in init_files:
    with open(init_file, 'a'):
        pass  # Just create empty file
