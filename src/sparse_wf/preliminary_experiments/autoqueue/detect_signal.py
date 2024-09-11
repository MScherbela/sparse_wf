import signal

global ABORT_CALCULATION
ABORT_CALCULATION = False

def signal_handler(signum, frame):
    global ABORT_CALCULATION
    print(f"Received signal {signum}")
    if signum == signal.SIGUSR1:
        print("Aborting calculation")
        ABORT_CALCULATION = True

signal.signal(signal.SIGUSR1, signal_handler)
