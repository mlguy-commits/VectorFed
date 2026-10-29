"""
Converts between Python objects (model weights + metadata)
and raw bytes for network transmission.

Wire format of a single packed message:
    [ 4-byte uint32: metadata length ][ JSON bytes ][ torch weights bytes ]

The outer 8-byte length header is handled by communication.py.
This file only knows about the inner structure.

Provides exactly two functions:
    pack(state_dict, metadata)   ->  bytes
    unpack(data)                 ->  (state_dict, metadata)
"""

import io 
import json
import struct
import time
import torch

# 4-byte unsigned int for inner metadata length header
METADATA_HEADER_FORMAT = '>I'
METADATA_HEADER_SIZE   = 4

def pack(state_dict: dict, metadata: dict) -> bytes:
    """
    Pack model weights and metadata into a single bytes object.

    Args:
        state_dict: PyTorch model state dict (from model.state_dict())
        metadata:   Dict containing fields:
                        client_id     (int)
                        round         (int)
                        num_samples   (int)
                        vector_clock  (list of ints)
                        timestamp     (float)  wall-clock time
                        msg_type      (str)    'broadcast' | 'update' | 'ack'
                        loss          (float)  optional, training loss
                        accuracy      (float)  optional, local accuracy
                        training_time (float)  optional, seconds spent training

    Returns:
        A bytes object ready to pass to send_message()

    Raises:
        TypeError:  If state_dict contains non-serializable tensors
        ValueError: If metadata is missing required fields
    """
    # Serialize metadata to JSON bytes
    metadata_bytes = json.dumps(metadata).encode('utf-8')

    # Serialize model weights to bytes using torch.save
    weights_buffer = io.BytesIO()
    torch.save(state_dict, weights_buffer)
    weights_bytes = weights_buffer.getvalue()

    # Pack inner header (4 bytes) = length of metadata
    metadata_length = len(metadata_bytes)
    inner_header    = struct.pack(METADATA_HEADER_FORMAT, metadata_length)

    # Concatenate everything
    # Layout: [4-byte header][JSON bytes][weights bytes]
    return inner_header + metadata_bytes + weights_bytes

def unpack(data: bytes) -> tuple:
    """
    Unpack bytes back into model weights and metadata.

    Args:
        data: Raw bytes received from recv_message()

    Returns:
        Tuple of (state_dict, metadata) where:
            state_dict  is a PyTorch state dict
            metadata    is a Python dict

    Raises:
        struct.error:       If header bytes are malformed
        json.JSONDecodeError: If metadata JSON is corrupted
        RuntimeError:       If weights bytes are corrupted
    """
    # Read inner header (first 4 bytes) to get metadata length
    inner_header    = data[:METADATA_HEADER_SIZE]
    (metadata_length,) = struct.unpack(METADATA_HEADER_FORMAT, inner_header)

    # Slice out metadata bytes and decode JSON
    metadata_start = METADATA_HEADER_SIZE
    metadata_end   = METADATA_HEADER_SIZE + metadata_length
    metadata_bytes = data[metadata_start:metadata_end]
    metadata       = json.loads(metadata_bytes.decode('utf-8'))

    # Remaining bytes are the model weights
    weights_bytes  = data[metadata_end:]
    weights_buffer = io.BytesIO(weights_bytes)
    state_dict     = torch.load(weights_buffer, weights_only=True)

    return state_dict, metadata

def make_metadata(
    msg_type:      str,
    client_id:     int   = -1,
    round_num:     int   = 0,
    num_samples:   int   = 0,
    vector_clock:  list  = None,
    loss:          float = 0.0,
    accuracy:      float = 0.0,
    training_time: float = 0.0,
) -> dict:
    
    """
    Helper to build a consistent metadata dict.
    Ensures all fields are always present with correct types.
    Avoids bugs from missing keys on the receiver side.

    Args:
        msg_type:      'broadcast' | 'update' | 'ack'
        client_id:     ID of the sending client (-1 for server)
        round_num:     Current communication round
        num_samples:   Number of training samples at client
        vector_clock:  Current vector clock state (list of ints)
        loss:          Final training loss this round
        accuracy:      Final local accuracy this round (0.0 to 100.0)
        training_time: Seconds spent on local training

    Returns:
        A dict with all fields populated
    """
    return {
        'msg_type':      msg_type,
        'client_id':     client_id,
        'round':         round_num,
        'num_samples':   num_samples,
        'vector_clock':  vector_clock if vector_clock is not None else [],
        'timestamp':     time.time(),   # wall-clock time (for SyncFed/OracleFed)
        'loss':          loss,
        'accuracy':      accuracy,
        'training_time': training_time,
    }