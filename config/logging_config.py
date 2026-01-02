"""
Logging configuration using Python dictConfig
More flexible than INI format
"""
import os
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },

    'filters': {
        'require_debug_false': {
            '()': 'logging.Filter',
        }
    },

    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'formatter': 'default',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': 'logs/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf-8'
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': 'logs/error.log',
            'maxBytes': 10485760,
            'backupCount': 5,
            'encoding': 'utf-8'
        },
        'json_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'json',
            'filename': 'logs/app.json',
            'maxBytes': 10485760,
            'backupCount': 5,
            'encoding': 'utf-8'
        }
    },

    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file', 'error_file'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn.access': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn.error': {
            'handlers': ['console', 'error_file'],
            'level': 'ERROR',
            'propagate': False
        },
        'fastapi': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'minio': {
            'handlers': ['file'],
            'level': 'WARNING',
            'propagate': False
        },
        'verification_service': {
            'handlers': ['console', 'file', 'json_file'],
            'level': 'DEBUG' if os.getenv('DEBUG') == 'true' else 'INFO',
            'propagate': False
        }
    }
}


def setup_logging():
    """Setup logging configuration"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    # Apply configuration
    logging.config.dictConfig(LOGGING_CONFIG)

    # Get root logger
    logger = logging.getLogger()
    logger.info("Logging configured successfully")

    return logger


# Convenience function
def get_logger(name: str):
    """Get a logger instance"""
    return logging.getLogger(name)
