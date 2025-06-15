import logging
import sys

from .encryption import encrypt, decrypt
from .merging import merge_datasets, sample_and_split_dataset
from .processing import encrypt_file, decrypt_file, encrypt_dataframe, decrypt_dataframe
from .huggingface import push_dataset_dict_to_hub

# Note: The `keys.py` module is not imported here because its only function,
# `derive_key`, is an internal implementation detail for the `encryption`
# module and is not intended to be part of the public `kushim.utils` API.

# Configure a default logger for the utils library.
# This prevents a "No handler found" warning if the consuming application
# doesn't configure logging. It can be easily overridden by the
# application's own logging configuration.
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

__all__ = [
    # encryption.py
    "encrypt",
    "decrypt",
    # merging.py
    "merge_datasets",
    "sample_and_split_dataset",
    # huggingface.py
    "push_dataset_dict_to_hub",
    # processing.py
    "encrypt_file",
    "decrypt_file",
    "encrypt_dataframe",
    "decrypt_dataframe",
] 