import threading
import queue


class LoggerThread(threading.Thread):
    def __init__(self, log_queue: queue.Queue[str], log_file: str) -> None:
        super().__init__(daemon=True)
        self.log_queue = log_queue
        self.log_file = log_file
        self.running = True

    def run(self) -> None:
        with open(self.log_file, "a") as f:
            while self.running:
                try:
                    entry = self.log_queue.get(timeout=0.1)
                    f.write(entry + "\n")
                    f.flush()
                except queue.Empty:
                    continue

    def stop(self) -> None:
        self.running = False


class DataSetLogger:
    """
    Encapsulates a thread which logs to a file.

    Args:
        log_file (str): File to log data to
    """

    def __init__(self, log_file: str) -> None:
        self.log_file = log_file
        self.log_queue = queue.Queue()
        self.logger = LoggerThread(self.log_queue, self.log_file)
        self.logger.start()

    def log(self, entry: str) -> None:
        self.log_queue.put(entry)

    def close(self) -> None:
        self.logger.stop()
        self.logger.join()

    def __del__(self) -> None:
        if self.logger.is_alive():
            self.close()
