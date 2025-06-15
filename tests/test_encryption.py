from kushim.utils.encryption import encrypt, decrypt

def test_encrypt_decrypt_symmetry():
    """
    Tests that encrypting a string and then decrypting it returns the
    original string. This is the fundamental contract of the functions.
    """
    original_text = "This is a secret message!"
    password = "a-strong-password"
    
    encrypted = encrypt(original_text, password)
    decrypted = decrypt(encrypted, password)
    
    assert decrypted == original_text
    assert encrypted != original_text

def test_decryption_with_wrong_password():
    """
    Tests that using the wrong password for decryption does not reveal the
    original text.
    """
    original_text = "Top secret data"
    correct_password = "password123"
    wrong_password = "password456"
    
    encrypted = encrypt(original_text, correct_password)
    decrypted = decrypt(encrypted, wrong_password)
    
    assert decrypted != original_text

def test_empty_string_handling():
    """
    Tests that the functions handle empty strings gracefully without errors.
    """
    password = "a-password"
    
    # Encrypting an empty string should result in an empty string
    assert encrypt("", password) == ""
    
    # Decrypting an empty string should result in an empty string
    assert decrypt("", password) == ""

def test_empty_password_handling():
    """
    Tests that the functions handle empty passwords gracefully, returning
    empty strings as a security measure.
    """
    original_text = "Some important text"
    
    # Encrypting with an empty password should fail gracefully
    assert encrypt(original_text, "") == ""
    
    # Decrypting with an empty password should also fail
    encrypted = encrypt(original_text, "real-password")
    assert decrypt(encrypted, "") == "" 