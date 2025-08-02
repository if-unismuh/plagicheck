#!/usr/bin/env python3
"""
Test Runner Script for Auto-Paraphrasing System
This script ensures proper Python path setup and runs the test suite.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """Set up the environment for running tests."""
    # Add the project root to Python path
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))
    
    # Set environment variables for testing
    os.environ.setdefault('TESTING', 'true')
    os.environ.setdefault('DATABASE_URL', 'sqlite:///./test.db')

def run_pytest(args):
    """Run pytest with the given arguments."""
    setup_environment()
    
    # Build pytest command
    cmd = [sys.executable, '-m', 'pytest'] + args
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Project root: {Path(__file__).parent.absolute()}")
    print(f"Python path: {sys.path[0]}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code: {e.returncode}")
        return e.returncode

def main():
    parser = argparse.ArgumentParser(description='Run tests for Auto-Paraphrasing System')
    parser.add_argument('args', nargs='*', help='Arguments to pass to pytest')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--performance', action='store_true', help='Run only performance tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    pytest_args = []
    
    if args.verbose:
        pytest_args.append('-v')
    
    if args.unit:
        pytest_args.extend(['-m', 'unit'])
    elif args.integration:
        pytest_args.extend(['-m', 'integration'])
    elif args.performance:
        pytest_args.extend(['-m', 'performance'])
    elif args.all:
        pytest_args.extend(['-m', 'not slow'])
    else:
        # Default: run all tests except slow ones
        pytest_args.extend(['-m', 'not slow'])
    
    # Add any additional args
    pytest_args.extend(args.args)
    
    return run_pytest(pytest_args)

if __name__ == '__main__':
    sys.exit(main()) 