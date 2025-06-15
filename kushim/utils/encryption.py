import base64
import logging

from .keys import derive_key

# Use the logger configured in the package's __init__.py
log = logging.getLogger(__name__)

def encrypt(plaintext: str, password: str) -> str:
    """
    Encrypt text with a XOR cipher derived from a password. The output is
    base64-encoded to ensure it's safe for transport and storage.
    """
    # Ensure plaintext is a string to handle various data types like numbers.
    plaintext = str(plaintext)
    
    if not plaintext or not password:
        return ""
    key = derive_key(password, len(plaintext))
    encrypted_bytes = bytes(a ^ b for a, b in zip(plaintext.encode(), key))
    return base64.b64encode(encrypted_bytes).decode()

def decrypt(ciphertext_b64: str, password: str) -> str:
    """
    Decrypt base64-encoded ciphertext that was encrypted with a XOR cipher.

    This is the counterpart to the encrypt function and is used to read
    encrypted data, for example from the BrowseComp dataset.
    """
    # Ensure input is a string to gracefully handle attempts to decrypt non-string data.
    ciphertext_b64 = str(ciphertext_b64)

    if not ciphertext_b64 or not password:
        return ""
    try:
        encrypted = base64.b64decode(ciphertext_b64)
        key = derive_key(password, len(encrypted))
        decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
        return decrypted.decode(errors='ignore')
    except Exception as e:
        # If decryption fails, log a warning and return the original value.
        # This prevents a single corrupt entry from halting the entire process.
        # The original (failed) value is returned to avoid data loss.
        log.warning(f"Decryption failed. Reason: {e}. Returning original value. Ciphertext (first 50 chars): '{ciphertext_b64[:50]}...'")
        return ciphertext_b64

def _xor_encrypt_decrypt(data: bytes, key: bytes) -> bytes:
    """Core XOR operation used for both encryption and decryption."""
    return bytes(a ^ b for a, b in zip(data, key)) 