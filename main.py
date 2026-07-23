"""
PriceIQ Pro V5 — Main Entry Point
This imports the advanced main file
"""
from main_advanced import app
from main_advanced import lifespan

# Re-export for compatibility
__all__ = ["app", "lifespan"]
