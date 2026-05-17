example_tiff_files = {}
example_tiff_files['1_channel_meso3'] = '/nearline/spruston/Arco_imaging/LICONN_mega_stack/2.5px_per_micron/file_00003_00001.tif'
example_tiff_files['1_channel_meso2'] = '/nearline/spruston/Arco_imaging/AB014/2024_11_05/3/AB014_2024_11_05_3__00001_00001.tif'
example_tiff_files['4_channel_meso3'] = '/nearline/spruston/Arco_imaging/AB34/2025_12_04/2/AB34_2025_12_04_2__00001_00001.tif'

# test one: time GappedMemmap(example_tiff_files['4_channel_meso3'])[::2,1]

from lazy_tiff_reader import gapped_memmap

def test_can_read_fancy_indices():
    for k, v in example_tiff_files.items():
        gm = gapped_memmap.GappedMemmap(v)
        assert len(gm.shape) == 4

        # Test indexing with valid indices for this file
        gm[1]  # Single frame
        gm[1, 0]  # Frame 1, channel 0 (always valid)
        gm[1, 0, 1]  # Frame 1, channel 0, row 1
        gm[1, 0, 1, 1]  # Frame 1, channel 0, pixel (1,1)
        gm[1:2]  # Frame slice
        gm[1, 0, 1:2]  # Frame 1, channel 0, row slice
        gm[[1, 3, 5], 0, [1, 3, 5], [1, 3, 5]]  # Fancy indexing

def test_io_efficiency():
    """Test IO efficiency when accessing a single pixel across all frames

    Current implementation loads entire frames even when only 1 pixel is needed.
    This test measures and reports the efficiency, expecting it to be very low.
    """
    # Use single-channel file for testing
    path = example_tiff_files['1_channel_meso2']

    gm = gapped_memmap.GappedMemmap(path, method='series')

    # Reset IO counter
    gm._bytes_read = 0

    # Access one pixel from each frame - very inefficient with current implementation
    # This loads entire frames but only uses 1 pixel each
    arr = gm[:, 0, 0, 0]

    # Calculate useful data size
    dtype_bytes = gm.dtype.itemsize
    useful_bytes = len(arr) * dtype_bytes

    # Get actual bytes read from disk
    actual_bytes = gm._bytes_read

    # Calculate efficiency
    io_efficiency = useful_bytes / actual_bytes if actual_bytes > 0 else 0

    # Report the metrics
    print(f"\nIO Efficiency Test Results:")
    print(f"  Useful data: {useful_bytes:,} bytes")
    print(f"  Actual bytes read: {actual_bytes:,} bytes")
    print(f"  IO efficiency: {io_efficiency:.10f} ({io_efficiency * 100:.6f}%)")
    print(f"  Overhead factor: {actual_bytes / useful_bytes if useful_bytes > 0 else 0:.1f}x")

    # This assertion will FAIL with current implementation - that's expected
    # We're measuring the problem, not fixing it yet
    assert io_efficiency > 0.5, f"IO efficiency too low: {io_efficiency:.6f} (expected > 0.5)"

    gm.close()
    
