import uuid

from module import logging_config


def test_setup_logger_disables_propagation_for_named_loggers():
    logger = logging_config.setup_logger(f"test_logger_{uuid.uuid4().hex}")

    try:
        assert logger.propagate is False
    finally:
        logger.handlers.clear()
