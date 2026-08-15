"""LLM-powered features: recommendation narration, RAG similar-restaurant
retrieval, and the Food Concierge chat copilot.

All modules are designed with graceful fallback to rules-based behavior
when the Anthropic API key is not configured — the app never breaks
without a key.
"""
