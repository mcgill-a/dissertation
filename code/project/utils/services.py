from datetime import datetime as dt


def timestamp():
    return dt.now().strftime("%Y-%m-%d %H:%M:%S")