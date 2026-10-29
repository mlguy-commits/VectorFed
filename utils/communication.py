"""
Handles all raw socket communication between server and clients.
Provides exactly two functions:
    send_message(sock, data)  ->  None
    recv_message(sock)        ->  bytes

Uses length-prefix framing:
    [ 8-byte header (message length) ][ N-byte payload ]

Nothing in this file knows about model weights, vector clocks,
or federated learning. It only moves bytes reliably.
"""

import struct

# Size of the length header in bytes (8 bytes = uint64 = max 18 exabytes)
HEADER_SIZE = 8

# How many bytes we read from the socket at a time
CHUNK_SIZE = 4096

def send_message(sock, data: bytes) -> None:
    """
    Send a complete message over a socket using length-prefix framing.

    Prepends an 8-byte big-endian header containing the length of data,
    then sends header + data atomically using sendall().

    Args:
        sock:  A connected socket object
        data:  The raw bytes to send

    Raises:
        OSError: If the socket is broken or the send fails
    """
    # Pack the length of data as an 8-byte big-endian unsigned integer
    header = struct.pack('>Q', len(data))

    sock.sendall(header + data)

def recv_message(sock) -> bytes:
    """
    Receive a complete message from a socket using length-prefix framing.

    First reads exactly 8 bytes to get the message length,
    then reads exactly that many bytes in chunks.

    Args:
        sock:  A connected socket object

    Returns:
        The complete message as bytes

    Raises:
        ConnectionError: If the connection drops mid-transfer
        struct.error:    If the header is malformed (< 8 bytes received)
    """
    # Read exactly 8 bytes (the header)
    header = _recv_exactly(sock, HEADER_SIZE)

    # Unpack header to get expected payload length
    (message_length,) = struct.unpack('>Q', header)

    # Read exactly message_length bytes (the payload)
    data = _recv_exactly(sock, message_length)

    return data

def _recv_exactly(sock, num_bytes: int) -> bytes:
    """
    Read exactly num_bytes from a socket, blocking until complete.

    TCP is a stream protocol. A single recv() call may return
    fewer bytes than requested. This function loops until
    exactly num_bytes have been received.

    Args:
        sock:       A connected socket object
        num_bytes:  Exactly how many bytes to read

    Returns:
        Exactly num_bytes as a bytes object

    Raises:
        ConnectionError: If the connection closes before num_bytes received
    """
    chunks = []
    bytes_received = 0

    while bytes_received < num_bytes:
        remaining = num_bytes - bytes_received
        chunk = sock.recv(min(CHUNK_SIZE, remaining))

        # recv() returns b'' (empty bytes) when the connection is closed
        if chunk == b'':
            raise ConnectionError(
                f"Connection closed after {bytes_received}/{num_bytes} bytes"
            )

        chunks.append(chunk)
        bytes_received += len(chunk)

    # Join all chunks into one contiguous bytes object
    return b''.join(chunks)