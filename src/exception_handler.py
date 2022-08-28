"""
@author: Jithin Sasikumar

Defines user-defined custom exception handlers. This provides an
interface with the flexibiity to add and define our own custom
exception handlers which can be used throughout the application.
This avoids any confusion and restricts direct changes to the
respective code files.

Note:
    Here, ValueError is a custom exception handling class and it is
    not similar to the standard built-in ValueError from python.
"""

class MLFlowError(Exception):
    pass

class ValueError(Exception):
    pass

class DirectoryError(Exception):
    pass

class NotFoundError(Exception):
    pass