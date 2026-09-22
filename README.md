# ERST: Learning Interaction from Entity Replacement in Spatiotemporal Forecasting

Anonymous implementation for reproducibility.

This repository contains the **full ERST model only**. Development-only experimental branches are not included. The implementation follows the paper terminology: an entity representation is replaced by a reference, the same function is evaluated before and after replacement, and the resulting difference is used as the response for that entity.

## Environment

```bash
pip install -r requirements.txt
```

The code uses PyTorch, NumPy, Pandas, and PyTables (`tables`, required for LargeST HDF5 files).

## Supported datasets

### Grid datasets

- NYCTaxi — 6 -> 1, 30 min, inflow, 7:1:2 split, metric/loss mask >= 10
- CHIBike — 6 -> 1, 30 min, outflow, 7:1:2 split, metric/loss mask >= 5
- T-Drive — 6 -> 1, 60 min, inflow, 7:1:2 split, metric/loss mask >= 10

### Node datasets

- PEMS03, PEMS04, PEMS07, PEMS08
- KnowAir
- SDWPF
- Milan SMS, Call, Internet
- LargeST CA/GBA/GLA/SD (2019 and 2021 presets)

## Expected data layout

The default preset paths are:

```text
data/
├── NYCTaxi/NYCTaxi.grid
├── CHIBike/CHIBike.grid
├── T-Drive/T-Drive.grid
├── PEMS/
│   ├── PEMS03.npz
│   ├── PEMS04.npz
│   ├── PEMS07.npz
│   └── PEMS08.npz
├── KnowAir/KnowAir_PM25.npy
├── SDWPF/SDWPF_Patv.npy
├── Milan/
│   ├── milan_400_sms.npy
│   ├── milan_400_call.npy
│   └── milan_400_internet.npy
└── LargeST/
    ├── ca/ca_his_2019.h5
    ├── ca/ca_his_2021.h5
    ├── gba/gba_his_2019.h5
    ├── gba/gba_his_2021.h5
    ├── gla/gla_his_2019.h5
    ├── gla/gla_his_2021.h5
    ├── sd/sd_his_2019.h5
    └── sd/sd_his_2021.h5
```

You can also override the path for a single run with `--data_path`.

## Run examples

```bash
# PEMS08
python erst.py --dataset PEMS08 --data_path ./data/PEMS/PEMS08.npz

# NYCTaxi
python erst.py --dataset NYCTaxi --data_path ./data/NYCTaxi/NYCTaxi.grid

# All three grid datasets using the default data layout
python erst.py --run_all_grid --data_root ./data

# All node-dataset presets using the default data layout
python erst.py --run_all_node --data_root ./data
```

The default training configuration is `D=64`, two interaction layers, dropout `0.1`, AdamW with learning rate `1e-3`, weight decay `1e-4`, early stopping patience `15`, and seed `42`.

## Data distribution

Raw and processed benchmark data are intentionally excluded from this repository by `.gitignore`. Dataset licenses are independent of this code repository. Please obtain each dataset from its original/authorized source and follow the corresponding terms. See `DATASETS.md`.

## Anonymous release checklist

Before submission, make sure the repository contains no author names, personal paths, emails, institution names, commit metadata that reveals identity, or private cloud links. The provided `erst.py` uses only relative paths by default.
