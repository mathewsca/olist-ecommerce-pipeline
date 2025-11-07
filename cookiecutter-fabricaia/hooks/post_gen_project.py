import os
import shutil


def remove_path(path: str) -> None:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)


def main() -> None:
    include_api = "{{ cookiecutter.include_api }}".lower() == "y"
    include_airflow = "{{ cookiecutter.include_airflow }}".lower() == "y"
    include_notebooks = "{{ cookiecutter.include_notebooks }}".lower() == "y"

    # Remove optional components if not selected
    if not include_api:
        remove_path("src/api")

    if not include_airflow:
        remove_path("pipelines/dags")

    if not include_notebooks:
        remove_path("notebooks")

    # If deep learning not selected, nothing to remove, requirements handle it


if __name__ == "__main__":
    main()
