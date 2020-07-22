import logging
import sys

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)        
    
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logging.getLogger('matplotlib.font_manager').disabled = True
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger