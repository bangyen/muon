#!/usr/bin/env python3
"""
Test runner for the Muon Optimizer test suite.

This script runs all tests and provides a comprehensive report.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_pattern=None, verbose=False, coverage=False):
    """Run the test suite"""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add test directory
    cmd.append("tests/")
    
    # Add specific test pattern if provided
    if test_pattern:
        cmd.append(f"-k {test_pattern}")
    
    # Run tests
    print("=" * 60)
    print("MUON OPTIMIZER TEST SUITE")
    print("=" * 60)
    print(f"Running tests from: {project_root}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED!")
        print("=" * 60)
        return False


def run_specific_test_suite(suite_name):
    """Run a specific test suite"""
    suites = {
        "optimizer": "test_optimizer",
        "model": "test_model", 
        "dataset": "test_dataset",
        "grokking": "test_grokking",
        "integration": "test_integration",
        "all": None
    }
    
    if suite_name not in suites:
        print(f"Unknown test suite: {suite_name}")
        print(f"Available suites: {', '.join(suites.keys())}")
        return False
    
    pattern = suites[suite_name]
    return run_tests(test_pattern=pattern, verbose=True)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run Muon Optimizer tests")
    parser.add_argument(
        "--suite", 
        choices=["optimizer", "model", "dataset", "grokking", "integration", "all"],
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true", 
        help="Generate coverage report"
    )
    parser.add_argument(
        "--pattern", "-k",
        help="Run tests matching pattern"
    )
    
    args = parser.parse_args()
    
    if args.suite == "all":
        success = run_tests(test_pattern=args.pattern, verbose=args.verbose, coverage=args.coverage)
    else:
        success = run_specific_test_suite(args.suite)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
