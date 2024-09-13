import signal
import time
import os
import detect_signal

print("PID: ", os.getpid())

def opt_step(state):
    time.sleep(1)
    return state + 1

opt_state = 0
for step in range(1_000_000):
    opt_state = opt_step(opt_state)
    print("Abort: ", detect_signal.ABORT_CALCULATION)
    if detect_signal.ABORT_CALCULATION:
        print("In main loop, aborting calculation")
        break


