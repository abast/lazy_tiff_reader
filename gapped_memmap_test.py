example_tiff_files = {}
example_tiff_files['1_channel_meso3'] = '/nearline/spruston/Arco_imaging/LICONN_mega_stack/2.5px_per_micron/file_00003_00001.tif'
example_tiff_files['1_channel_meso2'] = '/nearline/spruston/Arco_imaging/AB014/2024_11_05/3/AB014_2024_11_05_3__00001_00001.tif'
example_tiff_files['4_channel_meso3'] = '/nearline/spruston/Arco_imaging/AB34/2025_12_04/2/AB34_2025_12_04_2__00001_00001.tif'

# test one: time GappedMemmap(example_tiff_files['4_channel_meso3'])[::2,1]

import gapped_memmap

def test_can_read_fancy_indices():
    for k,v in example_tiff_files:
        gm = gapped_memmap.GappedMemmap(path)
        assert len(gm.shape) == 4
        gm[1]
        gm[1,1]
        gm[1,1,1]
        gm[1,1,1,1]
        gm[1:2]
        gm[1,1,1:2]
        gm[[1,3,5],0,[1,3,5],[1,3,5]]

def test_io_efficiency():
    # need to profile how much data is read in. placeholder 'profile_io' needs to be replaced with something that works.
    gm = gapped_memmap.GappedMemmap(path)
    with profile_io:
        # complicated to read ... goes through all slices. 
        arr = gm[:,0,0,0]
    # assume profile_io now holds the number of byts that were read in the above calls
    # determine how much data we have
    dtype_bytes = ... # todo, how many bytes is it per element in arr?
    arr_size = len(arr) * dtype_bytes
    
    io_efficiency = arr_size / profile_io
    assert io_efficiency > 0.5
    
