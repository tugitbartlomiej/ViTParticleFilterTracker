import ctypes, os, time, json
from pathlib import Path
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.FreeConsole()
if not k.AttachConsole(17864):
    raise ctypes.WinError(ctypes.get_last_error())
try:
    k.SetConsoleCtrlHandler(None,True)
    ids=(ctypes.c_uint32*32)()
    count=k.GetConsoleProcessList(ids,32)
    pids=list(ids)[:count]
    print('CONSOLE_PIDS',pids,flush=True)
    if count>32 or set(pids)-{17864,os.getpid()}:
        raise RuntimeError('Refuse interrupt: console contains unrelated processes')
    if not k.GenerateConsoleCtrlEvent(0,0):
        raise ctypes.WinError(ctypes.get_last_error())
    time.sleep(2)
    print('SCOPED_CTRL_C_SENT',flush=True)
finally:
    k.FreeConsole()
