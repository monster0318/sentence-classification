
import time
from functools import wraps

def timing(f):
    """
    timing function decorator
    usage: 
    @timing
    def function(a):
       pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print("time:%r taken: %2.5f sec" % (f.__name__, end - start))
        return result
    return wrapper