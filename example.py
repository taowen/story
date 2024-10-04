from step import step
from run_flow import run_flow
import time

@step
def example_function(a: int, b: int) -> int:
    time.sleep(10)
    return a + b

def worker_thread():
    example_function(1, 2)

if __name__ == "__main__":
    run_flow(worker_thread)
