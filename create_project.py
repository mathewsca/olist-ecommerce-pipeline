#!/usr/bin/env python3
"""
FabricaIA CookieCutter Template

This script creates a new FabricaIA project using the cookiecutter template.
"""

import os
import subprocess
import sys
from pathlib import Path


def install_cookiecutter():
    """Install cookiecutter if not already installed."""
    try:
        import cookiecutter
    except ImportError:
        print("Installing cookiecutter...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cookiecutter"])


def create_project():
    """Create a new FabricaIA project."""
    install_cookiecutter()
    
    # Get project details from user
    project_name = input("Enter project name: ").strip()
    if not project_name:
        project_name = "my_fabricaia_project"
    
    author_name = input("Enter author name: ").strip()
    if not author_name:
        author_name = "Data Scientist"
    
    author_email = input("Enter author email: ").strip()
    if not author_email:
        author_email = "author@example.com"
    
    # Create project using cookiecutter
    template_path = Path(__file__).parent / "cookiecutter-fabricaia"
    
    cmd = [
        "cookiecutter",
        str(template_path),
        "--no-input",
        f"project_name={project_name}",
        f"author_name={author_name}",
        f"author_email={author_email}",
        "python_version=3.11",
        "include_deep_learning=y",
        "include_api=y",
        "include_airflow=y",
        "include_notebooks=y"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Project '{project_name}' created successfully!")
        print(f"📁 Navigate to the project directory: cd {project_name}")
        print(f"🚀 Install dependencies: pip install -r requirements.txt")
        print(f"📊 Run example notebook: jupyter notebook notebooks/fabricaia_example.ipynb")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating project: {e}")
        sys.exit(1)


if __name__ == "__main__":
    create_project()
