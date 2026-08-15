"""Integration tests for Dabba.

Tests that exercise multiple components together:
- Pipeline → model persistence → API serving
- Feature engineering → model training → prediction
- Database reads → cache writes → cache reads

These tests are slower than unit tests and may require
pre-existing data or model artifacts.
"""
