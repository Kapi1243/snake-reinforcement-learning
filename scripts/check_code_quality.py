"""
Code quality check script.

Run all code quality tools: black, isort, flake8, mypy, bandit, pytest
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {description}...")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    success = result.returncode == 0
    
    if success:
        print(f"✅ {description} passed")
    else:
        print(f"❌ {description} failed")
    
    return success


def main():
    """Run all code quality checks."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    tests_dir = project_root / "tests"
    
    checks = [
        (
            ["black", "--check", str(src_dir), str(tests_dir)],
            "Black (code formatting)"
        ),
        (
            ["isort", "--check-only", str(src_dir), str(tests_dir)],
            "isort (import sorting)"
        ),
        (
            ["flake8", str(src_dir), str(tests_dir)],
            "Flake8 (linting)"
        ),
        (
            ["mypy", str(src_dir), "--ignore-missing-imports"],
            "MyPy (type checking)"
        ),
        (
            ["bandit", "-r", str(src_dir), "-ll"],
            "Bandit (security)"
        ),
        (
            ["pytest", str(tests_dir), "-v"],
            "PyTest (tests)"
        ),
    ]
    
    results = []
    for cmd, desc in checks:
        try:
            success = run_command(cmd, desc)
            results.append((desc, success))
        except FileNotFoundError:
            print(f"⚠️  {desc} tool not found. Skipping...")
            results.append((desc, None))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for desc, success in results:
        if success is None:
            status = "⚠️  SKIPPED"
        elif success:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{status}: {desc}")
    
    # Exit code
    if any(success is False for _, success in results):
        print("\n❌ Some checks failed!")
        sys.exit(1)
    else:
        print("\n✅ All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
