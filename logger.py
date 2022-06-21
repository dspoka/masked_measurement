import datetime
import os

class Logger:
    def __init__(self):
        self.secret = None

    def set(self, log_path=None, reopen_to_flush=True):
        self.log_file = None
        self.reopen_to_flush = reopen_to_flush
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, 'a+')

    def log(self, msg):
        formatted = '[{}] {}'.format(
            datetime.datetime.now().replace(microsecond=0).isoformat(),
            msg)
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + '\n')
            if self.reopen_to_flush:
                log_path = self.log_file.name
                self.log_file.close()
                self.log_file = open(log_path, 'a+')
            else:
                self.log_file.flush()

LOGGER = Logger()