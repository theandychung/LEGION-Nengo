from wmi import WMI
import os

def memory():
    #return memory usage
    w = WMI('.')
    result = w.query("SELECT WorkingSet FROM Win32_PerfRawData_PerfProc_Process WHERE IDProcess=%d" % os.getpid())
    return int(result[0].WorkingSet)
# https://stackoverflow.com/questions/938733/total-memory-used-by-python-process