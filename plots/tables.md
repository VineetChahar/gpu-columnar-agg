# Generated tables (scripts/plot_all.py)


## V5 streaming / PCIe wall

| query | mode | streams | chunk rows | ms | eff. PCIe GB/s | CPU engine ms | correct |
|---|---|---|---|---|---|---|---|
| groupby_card16 | cpu_engine_host_resident | 0 | 0 | 263.9 | 2.0 | 263.9 | True |
| groupby_card16 | gpu_resident_kernel_only | 1 | 0 | 16.1 | 32.2 | 263.9 | True |
| groupby_card16 | gpu_stream_pageable_overlap | 2 | 16000000 | 117.1 | 4.4 | 263.9 | True |
| groupby_card16 | gpu_stream_pageable_overlap | 3 | 16000000 | 118.2 | 4.4 | 263.9 | True |
| groupby_card16 | gpu_stream_pageable_overlap | 4 | 16000000 | 119.1 | 4.4 | 263.9 | True |
| groupby_card16 | gpu_stream_pageable_serial | 1 | 16000000 | 130.3 | 4.0 | 263.9 | True |
| groupby_card16 | gpu_stream_pinned_overlap | 2 | 16000000 | 43.1 | 12.0 | 263.9 | True |
| groupby_card16 | gpu_stream_pinned_overlap | 3 | 16000000 | 44.7 | 11.6 | 263.9 | True |
| groupby_card16 | gpu_stream_pinned_overlap | 4 | 16000000 | 45.7 | 11.3 | 263.9 | True |
| groupby_card16 | gpu_stream_pinned_overlap_chunk | 3 | 1048576 | 51.8 | 10.0 | 263.9 | True |
| groupby_card16 | gpu_stream_pinned_overlap_chunk | 3 | 4194304 | 47.7 | 10.9 | 263.9 | True |
| groupby_card16 | gpu_stream_pinned_overlap_chunk | 3 | 67108864 | 47.2 | 11.0 | 263.9 | True |
| groupby_card16 | gpu_stream_pinned_serial | 1 | 16000000 | 54.7 | 9.5 | 263.9 | True |
| groupby_card16 | host_register_cost |  |  | 41.2 | nan | nan |  |

## V6 join (ms; probed = rows that reached the hash table)

| D | sel | variant | ms | probed | bloom FPR | CPU ms | speedup |
|---|---|---|---|---|---|---|---|
| 1,048,576 | 1 | v6_dict | 2.39 | 1,048,576 | nan | 440.8 | 184.5 |
| 1,048,576 | 1 | v6_np | 16.65 | 20,000,000 | nan | 440.8 | 26.5 |
| 1,048,576 | 1 | v6_np_bloom | 10.53 | 204,606 | 0.0000 | 440.8 | 41.9 |
| 1,048,576 | 1 | v6_part | 21.70 | 20,000,000 | nan | 440.8 | 20.3 |
| 1,048,576 | 1 | v6_part_bloom | 21.15 | 204,606 | 0.0000 | 440.8 | 20.8 |
| 1,048,576 | 100 | v6_dict | 6.36 | 1,048,576 | nan | 777.8 | 122.2 |
| 1,048,576 | 100 | v6_np | 25.19 | 20,000,000 | nan | 777.8 | 30.9 |
| 1,048,576 | 100 | v6_np_bloom | 33.46 | 20,000,000 | nan | 777.8 | 23.2 |
| 1,048,576 | 100 | v6_part | 68.39 | 20,000,000 | nan | 777.8 | 11.4 |
| 1,048,576 | 100 | v6_part_bloom | 76.53 | 20,000,000 | nan | 777.8 | 10.2 |
| 16,777,216 | 1 | v6_dict | 19.30 | 16,777,216 | nan | 1574.5 | 81.6 |
| 16,777,216 | 1 | v6_np | 22.83 | 20,000,000 | nan | 1574.5 | 69.0 |
| 16,777,216 | 1 | v6_np_bloom | 22.41 | 200,708 | 0.0000 | 1574.5 | 70.2 |
| 16,777,216 | 1 | v6_part | 54.00 | 20,000,000 | nan | 1574.5 | 29.2 |
| 16,777,216 | 1 | v6_part_bloom | 42.49 | 200,708 | 0.0000 | 1574.5 | 37.1 |
| 16,777,216 | 100 | v6_dict | 81.80 | 16,777,216 | nan | 2634.0 | 32.2 |
| 16,777,216 | 100 | v6_np | 86.33 | 20,000,000 | nan | 2634.0 | 30.5 |
| 16,777,216 | 100 | v6_np_bloom | 122.24 | 20,000,000 | nan | 2634.0 | 21.5 |
| 16,777,216 | 100 | v6_part | 129.97 | 20,000,000 | nan | 2634.0 | 20.3 |
| 16,777,216 | 100 | v6_part_bloom | 169.90 | 20,000,000 | nan | 2634.0 | 15.5 |

## nsys (3-stream V5 pipeline)

kernel time hidden under H2D copies: 93.2%; H2D time with a kernel running concurrently: 26.6%
