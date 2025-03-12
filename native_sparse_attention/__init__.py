# -*- coding: utf-8 -*-

from .ops.naive import naive_nsa, naive_nsa_with_compression
from .ops.parallel import parallel_nsa, parallel_nsa_with_compression

__all__ = [
    'naive_nsa',
    'parallel_nsa',
    'naive_nsa_with_compression',
    'parallel_nsa_with_compression'
]


__version__ = '0.0'