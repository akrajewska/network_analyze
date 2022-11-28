import os
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import normalize
from torch.nn.functional import normalize as normalize_torch

DATA_DIR = Path('DATA')
QUANT_COLS = ['Flow Duration',
              'Total Fwd Packets', 'Total Backward Packets',
              'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
              'Fwd Packet Length Max', 'Fwd Packet Length Min',
              'Fwd Packet Length Mean', 'Fwd Packet Length Std',
              'Bwd Packet Length Max', 'Bwd Packet Length Min',
              'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
              'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
              'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
              'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
              'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
              'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
              'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
              'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
              'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
              'PSH Flag Count', 'ACK Flag Count',
              'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
              'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
              'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
              'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
              'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
              'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean',
              'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
              'Idle Std', 'Idle Max', 'Idle Min']

LABELS = {
    'Bot':1,
    'DDoS':2,
    'DoS GoldenEye':3,
    'DoS Hulk':4,
    'DoS Slowhttptest': 5,
    'DoS slowloris': 6,
    'FTP-Patator': 7,
    'Heartbleed': 8,
    'PortScan': 9,
    'SSH-Patator': 10,
    'Web Attack  Brute Force': 11,
    'Web Attack   Sql Injection':12,
    'Web Attack   XSS': 13
}
def prepare_df():
    # df = pd.read_csv(dfile, parse_dates=[' Timestamp'], date_parser=dateparse)
    df = pd.concat((pd.read_csv(DATA_DIR / f) for f in os.listdir(DATA_DIR)), ignore_index=True)
    df = df.rename(columns=lambda x: x.strip())
    return df


def data_arrays(df):
    df = df[QUANT_COLS + ['Label']]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[df.select_dtypes(include=[np.number]).ge(0).all(1)]
    df = df.dropna()
    X = df.to_numpy()
    y = X[:, -1]
    X = X[:, :-1]
    # normalizuje wierszami
    X = normalize(X)
    return X, y


def data_tensors(df):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    y = torch.tensor(df['Label'].map(LABELS).values,  device=device)
    df = df[QUANT_COLS]
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df = df[df.select_dtypes(include=[np.number]).ge(0).all(1)]
    # df = df.dropna()
    #X = torch.from_numpy(df.values, dtype=torch.float32, device=device)
    X = torch.tensor(df.values, dtype=torch.float32, device=device)

    # X = X[:, :-1]
    X = normalize_torch(X)
    return X, y