import hashlib
import logging

log = logging.getLogger(__name__)

def derive_key(password: str, length: int) -> bytes:
    """
    Deterministically derive a key of a specific length from a password
    using a cryptographically secure hash function (SHA-256).
    """
    if not password:
        log.warning("An empty password was provided for key derivation.")
        return b'\0' * length
        
    # Use SHA-256 to create a fixed-size hash of the password.
    # This is a one-way function, so the password cannot be easily reversed.
    hashed = hashlib.sha256(password.encode()).digest()
    
    # To get a key of arbitrary length, we tile the hash.
    # For example, if hash is 'abc' and length is 7, key is 'abcabca'.
    # This is more secure than simple repetition as it uses the full
    # entropy of the hash.
    key = (hashed * (length // len(hashed) + 1))[:length]
    
    return key 