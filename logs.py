import logging
import os

logger = logging.getLogger('RAG-APP')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S %p'))
logger.addHandler(console_handler)
