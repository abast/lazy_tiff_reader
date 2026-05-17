import ctypes
import numpy as np
import os


def copy_byte_spans_between_buffers(buffer_in, buffer_out, offsets_in, offsets_out, lengths):
    src = buffer_in.ctypes.data
    dst = buffer_out.ctypes.data

    for oi, oo, l in zip(offsets_in, offsets_out, lengths):
        # Debug: print types
        if not isinstance(oi, (int, np.integer)):
            print(f"WARNING: oi type = {type(oi)}, value = {oi}")
        if not isinstance(oo, (int, np.integer)):
            print(f"WARNING: oo type = {type(oo)}, value = {oo}")
        if not isinstance(l, (int, np.integer)):
            print(f"WARNING: l type = {type(l)}, value = {l}")

        # Convert to int to ensure proper types
        oi = int(oi)
        oo = int(oo)
        l = int(l)

        ctypes.memmove(dst + oo, src + oi, l)


def _get_example_arrays(test_data_folder_path):
    '''creates test arrays'''
    out = {}
    arr = np.arange(100, dtype=np.uint8)
    path = os.path.join(test_data_folder_path, 'array')
    with open(path, 'wb') as f:
        f.write(arr.tobytes())

    out['in_memory'] = arr
    out['memmap'] = np.memmap(path, dtype=np.uint8, mode="r")
    return out


def copy_byte_spans_between_buffers_test():
    # uint8 → one element = one byte
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_folder:
        arrays = _get_example_arrays(tmp_folder)

        for mode, A in arrays.items():
            B = np.zeros(50, dtype=np.uint8)

            offsets_in  = [0, 75]
            offsets_out = [0, 25]
            lengths     = [25, 25]

            copy_byte_spans_between_buffers(A, B, offsets_in, offsets_out, lengths)

            expected = np.concatenate([A[0:25], A[75:100]])
            assert np.array_equal(B, expected)

if __name__ == '__main__':
    print('running tests')
    copy_byte_spans_between_buffers_test()
