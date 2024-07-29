# Path: llmops/config.py
"""
This module contains the configuration for the LLMOPS package.

It determines the execution type of the package.
It contains the following configuration:
- EXECUTION_TYPE: The type of execution for the package.
  Valid values: LOCAL, AZURE
  If environment variable LLMOPS_EXECUTION_TYPE is set,
    its value is used instead.
"""

import os

try:
    EXECUTION_TYPE = os.environ["LLMOPS_EXECUTION_TYPE"]
    print(
      "Setting execution type from environment variable:",
      EXECUTION_TYPE
    )
except KeyError:
    EXECUTION_TYPE = "LOCAL"  # You can edit this value (LOCAL, AZURE)
    print(
      "Setting execution type from llmops module configuration:",
      EXECUTION_TYPE
    )
