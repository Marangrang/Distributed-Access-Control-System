"""Script to automatically fix common flake8 issues."""
from pathlib import Path


def fix_file(filepath):
    """Fix common linting issues in a Python file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Remove trailing whitespace
    lines = [line.rstrip() + '\n' for line in lines]

    # Remove blank lines with whitespace
    lines = ['\n' if line.strip() == '' else line for line in lines]

    # Ensure file ends with newline
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'

    # Remove trailing blank lines except one
    while len(lines) > 1 and lines[-1] == '\n' and lines[-2] == '\n':
        lines.pop()

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Fixed: {filepath}")


def main():
    """Fix all Python files in the project."""
    base_dir = Path(__file__).parent

    python_files = [
        'cloud_ingestion/ingest.py',
        'config/gunicorn.conf.py',
        'edge_sync/sync.py',
        'serve/__init__.py',
        'serve/main.py',
        'serve/run.py',
        'serve/wsgi.py',
        'train/__init__.py',
        'train/train.py',
        'train/upload_index_to_minio.py',
        'verification_service/__init__.py',
        'verification_service/build_index.py',
        'verification_service/main.py',
        'verification_service/metrics.py',
        'verification_service/test_verify.py',
        'tests/test_api.py',
        'tests/test_database.py',
        'tests/test_minio.py',
        'tests/test_face_verification.py',
        'tests/test_integration.py',
    ]

    for file in python_files:
        filepath = base_dir / file
        if filepath.exists():
            fix_file(filepath)


if __name__ == '__main__':
    main()
