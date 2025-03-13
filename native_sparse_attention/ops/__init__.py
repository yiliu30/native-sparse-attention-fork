# -*- coding: utf-8 -*-

from .naive import naive_nsa, naive_nsa_with_compression
from .parallel import parallel_nsa, parallel_nsa_with_compression

__all__ = [
    'naive_nsa',
    'parallel_nsa',
    'naive_nsa_with_compression',
    'parallel_nsa_with_compression'
]
