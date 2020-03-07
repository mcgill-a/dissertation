from datetime import datetime as dt


def timestamp():
    print(dt.now().strftime("%Y-%m-%d %H:%M:%S"))