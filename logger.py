import os
import time
import logging


def create_logger(logname):
    os.makedirs('logs', exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = os.path.join('logs', f"{logname}_{rq}.log")
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger
