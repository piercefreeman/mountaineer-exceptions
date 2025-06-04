import logging

import pytest
from mountaineer.logging import LOGGER


@pytest.fixture(autouse=True)
def set_logging_level():
    """Set logging level to INFO for the mountaineer_exceptions package"""
    LOGGER.getChild("mountaineer_exceptions").setLevel(logging.INFO)
