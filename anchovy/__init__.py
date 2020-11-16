__release__ = '0.0.0.dev0'
__version__ = '0.0'


import logging

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

info_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_formatter = logging.Formatter(
    '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s:%(message)s')
ch.setFormatter(info_formatter)

# initialize package-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)  # set critical logging level
logger.addHandler(ch)


def set_logging_level(level):
    if level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif level == 'info':
        logger.setLevel(logging.INFO)
    elif level == 'error':
        logger.setLevel(logging.ERROR)
    elif level == 'warning':
        logger.setLevel(logging.WARNING)
    elif level == 'critical':
        logger.setLevel(logging.CRITICAL)
    else:
        raise ValueError("Unknown logging level :{}".format(level))

    if level == 'debug':
        ch.setFormatter(debug_formatter)
    else:
        ch.setFormatter(info_formatter)
