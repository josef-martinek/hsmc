import logging


def create_custom_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers to avoid adding multiple handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('{levelname}: {message}', style='{')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Optional: Avoid propagation to root logger
        logger.propagate = False

    return logger