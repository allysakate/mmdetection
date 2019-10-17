# Python Program to Convert seconds 
# into hours, minutes and seconds 
import datetime 
import time

class Timer(object):
    def __init__(self, description):
        self.description = description
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, type, value, traceback):
        self.end = time.time()
        interval =  self.end - self.start 
        self.convert = datetime.timedelta(seconds = interval)
        print(f"{self.description}: {self.convert}")