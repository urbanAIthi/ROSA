import logging
from functools import wraps

def logger(func):
    """
    Decorator that attaches a logger to the decorated function based on its module.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The wrapped function with a `logger` attribute attached.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        setattr(wrapped, 'logger', logger)  # Setting the logger to the function
        return func(*args, **kwargs)
    return wrapped
