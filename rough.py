def utf8span(s: str, pos: int):
    """
    Compute the utf8 character index of the [pos] position within the string s
    """

    return len(s[:pos].encode('utf-8'))


def reverse_utf8span(s: str, byte_index: int) -> int:
    """
    Compute the string character index of the [byte_index] position within the UTF-8 encoded string s
    """
    encoded = s.encode('utf-8')
    truncated_encoded = encoded[:byte_index]
    return len(truncated_encoded.decode('utf-8', errors='ignore'))


a = "楽天モバイルで楽天クレジットカードが必要"
byte_index = utf8span(a,9)

char_index = reverse_utf8span(a,byte_index=byte_index)

print(byte_index)
print(char_index)
