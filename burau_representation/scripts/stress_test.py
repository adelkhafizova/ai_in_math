import multiprocessing
import time

# Function to run in each process (will just run an infinite loop)
def cpu_stress():
    while True:
        pass  # The busy-wait loop that will stress the CPU

if __name__ == '__main__':
    # Get the number of physical cores in the system
    num_processes = multiprocessing.cpu_count()  # Number of cores (logical CPUs)

    # Create and start one process per core
    processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=cpu_stress)
        p.start()
        processes.append(p)

    # Optional: Keep the main process alive while child processes are running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # When you press Ctrl+C, terminate all the processes
        for p in processes:
            p.terminate()
        print("Test finished.")