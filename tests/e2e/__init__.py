"""End-to-end tests for Dabba.

Tests that simulate real user workflows:
- Full pipeline run → dashboard loads with correct data
- API request → response → cache populated
- Chat message → concierge response flows correctly

These tests require a fully configured environment
with trained models and may take minutes to run.
"""
