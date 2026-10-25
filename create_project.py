from pathlib import Path

#current working directory
root=Path(".").resolve()

#folders for the project 
folders=[
    "app/api",
    "app/core",
    "app/rag",
    "app/services",
    "static",
    "templates",
    "data"   
]

#files to be created in the project
files=[
    "app/__init__.py",
    "app/main.py",
    "app/services/ingestion.py",
    "app/api/__init__.py",
    "app/core/__init__.py",
    "app/rag/__init__.py",
    "app/services/__init__.py",
    ".env",
    # "requirements.txt",
    "run.py"
]

#create folder 
for folder in folders:
    folder_path=root/folder
    folder_path.mkdir(parents=True,exist_ok=True)
    print(f"Folder '{folder}' created successfully.")

#create files
for file in files:
    file_path=root/file
    file_path.touch(exist_ok=True)
    print(f"File '{file}' created successfully.")