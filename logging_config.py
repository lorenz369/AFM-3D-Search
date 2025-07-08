import logging
from logging.handlers import TimedRotatingFileHandler
import os

# Create a 'logs' directory if it doesn't exist
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)

# Define the format for log messages
LOG_FORMAT = '[%(asctime)s] [%(process)d] [%(levelname)s] [%(name)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S %z'

# Define the configuration dictionary
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': LOG_FORMAT,
            'datefmt': DATE_FORMAT,
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
        'file_rotator': {
            'level': 'DEBUG',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'formatter': 'standard',
            'filename': os.path.join(LOGS_DIR, 'app.log'),
            'when': 'midnight', # Rotate logs at midnight
            'backupCount': 7,  # Keep 7 days of backup logs
            'encoding': 'utf8',
        },
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file_rotator'],
            'level': 'DEBUG',
            'propagate': True,
        },
    }
}