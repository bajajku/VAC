# This file makes data_collection a package
from .json_parser import JsonParser

# Expose the class at package level for easier importing
__all__ = ['JsonParser'] 