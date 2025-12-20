import numpy as np


def slices_to_offsets(key, page_offsets, shape, nchannels, dtype_itemsize):
    """
    Convert array indexing key to byte offsets for memory-mapped file access.

    Parameters
    ----------
    key : int, slice, list, or tuple
        Indexing key from __getitem__ (can be any valid numpy index)
    page_offsets : np.ndarray
        Byte offsets for each page in the file
    shape : tuple
        Full array shape (nframes, nchannels, height, width)
    nchannels : int
        Number of channels
    dtype_itemsize : int
        Size of each element in bytes

    Returns
    -------
    offsets_in : tuple
        Byte offsets to read from source file
    offsets_out : tuple
        Byte offsets to write to destination buffer
    lengths : tuple
        Number of bytes to copy for each chunk
    """
    # Normalize key to 4-tuple
    normalized_key = _normalize_key(key, shape)

    # Expand to index lists
    frame_indices = _expand_key(normalized_key[0], shape[0])
    channel_indices = _expand_key(normalized_key[1], shape[1])
    y_indices = _expand_key(normalized_key[2], shape[2])
    x_indices = _expand_key(normalized_key[3], shape[3])

    # Compute chunks
    offsets_in_list = []
    offsets_out_list = []
    lengths_list = []

    dest_offset = 0
    width = shape[3]

    for frame_idx in frame_indices:
        for channel_idx in channel_indices:
            # Compute page index
            if nchannels == 1:
                page_idx = frame_idx
            else:
                page_idx = frame_idx * nchannels + channel_idx

            page_base_offset = page_offsets[page_idx]

            # Check if we can read contiguous rows
            if _is_contiguous_rows(y_indices, x_indices, width):
                # Optimized: read full rows
                y_start = y_indices[0]
                num_rows = len(y_indices)
                row_size_bytes = width * dtype_itemsize

                src_offset = page_base_offset + y_start * row_size_bytes
                length = num_rows * row_size_bytes

                offsets_in_list.append(src_offset)
                offsets_out_list.append(dest_offset)
                lengths_list.append(length)
                dest_offset += length
            else:
                # Pixel-by-pixel
                for y in y_indices:
                    for x in x_indices:
                        pixel_offset = (y * width + x) * dtype_itemsize
                        src_offset = page_base_offset + pixel_offset

                        offsets_in_list.append(src_offset)
                        offsets_out_list.append(dest_offset)
                        lengths_list.append(dtype_itemsize)
                        dest_offset += dtype_itemsize

    return tuple(offsets_in_list), tuple(offsets_out_list), tuple(lengths_list)


def _normalize_key(key, shape):
    """Convert any indexing key to 4-tuple of (frame, channel, y, x) keys."""
    # Handle scalar index
    if not isinstance(key, tuple):
        key = (key,)

    # Pad with : to make 4D
    ndim = len(shape)
    if len(key) < ndim:
        key = key + (slice(None),) * (ndim - len(key))

    return key


def _expand_key(key, size):
    """Convert a single dimension key (int/slice/list) to list of indices."""
    if isinstance(key, int):
        # Scalar index -> single element
        return [key]
    elif isinstance(key, slice):
        # Slice -> range of indices
        return list(range(*key.indices(size)))
    elif isinstance(key, (list, tuple, np.ndarray)):
        # Fancy indexing -> direct indices
        return list(key)
    else:
        raise TypeError(f"Invalid index type: {type(key)}")


def _is_contiguous_rows(y_indices, x_indices, width):
    """Check if accessing contiguous full-width rows."""
    # Must access all x indices
    if len(x_indices) != width or x_indices != list(range(width)):
        return False

    # Check if y is contiguous
    if len(y_indices) < 2:
        return True

    return all(y_indices[i+1] == y_indices[i] + 1 for i in range(len(y_indices) - 1))
