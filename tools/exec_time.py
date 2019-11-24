import time
import math

def get_proctime(start_time):
    t = time.time() - start_time
    t_msec, t_sec = math.modf(t)
    _, t_msec = math.modf(t_msec * 1000)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    proc_time = '{}hour:{}min:{}sec.{}msec'.format(t_hour,t_min,t_sec,t_msec)
    return proc_time
