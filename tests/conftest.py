import sys
import os
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock Pinecone and Gemini at module level before main.py is imported
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

pinecone_mock = patch("pinecone.Pinecone").start()
pinecone_mock.return_value.Index.return_value = MagicMock()

genai_mock = patch("google.genai.Client").start()
genai_mock.return_value = MagicMock()