"""
Pytest configuration and shared fixtures for Insurance RAG Assistant tests.
"""
import sys
import os

# Ensure the project root is in sys.path so main.py can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
