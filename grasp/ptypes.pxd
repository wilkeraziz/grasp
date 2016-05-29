"""
Primitive types: definitions.

:Authors: - Wilker Aziz
"""

ctypedef long int_t
ctypedef double real_t
ctypedef long id_t  # nodes/edges/states
cimport numpy as np
ctypedef np.double_t weight_t
ctypedef np.int8_t boolean_t
ctypedef np.int8_t status_t