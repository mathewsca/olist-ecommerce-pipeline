#!/usr/bin/env python3
"""
FabricaIA CookieCutter Template

This script creates a new FabricaIA project using the cookiecutter template.
"""

import importlib.util
import os
import subprocess
import sys
import warnings
from pathlib import Path

# Silence noisy third-party dependency warnings
warnings.filterwarnings("ignore", message="urllib3 .* doesn't match a supported version")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")


def ensure_environment():
    """Ensure cookiecutter is available, auto-switching to local ./venv if detected."""
    # Check if cookiecutter is already available in current interpreter
    if importlib.util.find_spec("cookiecutter") is not None:
        return

    # Check if a local virtualenv exists in the repository
    root_dir = Path(__file__).parent
    venv_python = root_dir / "venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = root_dir / "venv" / "Scripts" / "python.exe"

    if venv_python.exists() and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
        print(f"🔄 Redirecionando execução para o ambiente virtual: {venv_python}")
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

    # Try installing cookiecutter via pip
    try:
        print("Instalando 'cookiecutter'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cookiecutter"])
    except Exception as e:
        print(f"\n❌ Erro: Não foi possível executar o instalador automático ({e}).")
        print("💡 Dica: Crie e ative um ambiente virtual antes de criar o projeto:")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate   # Linux/macOS (ou venv\\Scripts\\activate no Windows)")
        print("   pip install -r requirements.txt")
        print("   python create_project.py\n")
        sys.exit(1)


def create_project():
    """Create a new FabricaIA project."""
    ensure_environment()

    # Get project details from user
    project_name = input("Enter project name [my_fabricaia_project]: ").strip()
    if not project_name:
        project_name = "my_fabricaia_project"

    author_name = input("Enter author name [Data Scientist]: ").strip()
    if not author_name:
        author_name = "Data Scientist"

    author_email = input("Enter author email [author@example.com]: ").strip()
    if not author_email:
        author_email = "author@example.com"

    # Optional heavy dependencies (defaults to 'n' for lightweight installation)
    include_dl_input = input("Include Deep Learning (PyTorch & TensorFlow)? [y/N]: ").strip().lower()
    include_dl = "y" if include_dl_input in ["y", "yes", "s", "sim"] else "n"

    include_airflow_input = input("Include Apache Airflow in local requirements? [y/N]: ").strip().lower()
    include_airflow = "y" if include_airflow_input in ["y", "yes", "s", "sim"] else "n"

    # Destination directory (defaults to parent directory if inside Template repository)
    is_inside_template = (Path.cwd() / "cookiecutter-fabricaia").is_dir()
    suggested_output_dir = ".." if is_inside_template else "."

    output_dir_input = input(
        f"Enter destination directory [{suggested_output_dir}]: "
    ).strip()
    if not output_dir_input:
        output_dir_input = suggested_output_dir

    output_dir = Path(output_dir_input).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create project using cookiecutter
    template_path = Path(__file__).parent / "cookiecutter-fabricaia"

    try:
        from cookiecutter.main import cookiecutter

        cookiecutter(
            str(template_path),
            output_dir=str(output_dir),
            no_input=True,
            extra_context={
                "project_name": project_name,
                "author_name": author_name,
                "author_email": author_email,
                "python_version": "3.11",
                "include_deep_learning": include_dl,
                "include_api": "y",
                "include_airflow": include_airflow,
                "include_notebooks": "y",
            },
        )
    except ImportError:
        cmd = [
            sys.executable,
            "-m",
            "cookiecutter",
            str(template_path),
            "--output-dir",
            str(output_dir),
            "--no-input",
            f"project_name={project_name}",
            f"author_name={author_name}",
            f"author_email={author_email}",
            "python_version=3.11",
            f"include_deep_learning={include_dl}",
            "include_api=y",
            f"include_airflow={include_airflow}",
            "include_notebooks=y",
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating project: {e}")
            sys.exit(1)

    final_path = output_dir / project_name
    print(f"\n✅ Project '{project_name}' created successfully at: {final_path}")
    print(f"\n📋 Próximos passos:")
    print(f"  1. Acesse o diretório do projeto:")
    print(f"     cd {final_path}")
    print(f"  2. Crie e ative o ambiente virtual:")
    print(f"     python3 -m venv venv")
    print(f"     source venv/bin/activate   # Linux/macOS (ou venv\\Scripts\\activate no Windows)")
    print(f"  3. Instale as dependências:")
    print(f"     make install               # Instalação padrão (ou make setup para dev completo)")
    print(f"  4. Execute o pipeline de teste:")
    print(f"     make data && make pipeline")
    print(f"  5. Inicie a API de inferência:")
    print(f"     make api")


if __name__ == "__main__":
    create_project()
