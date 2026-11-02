"""
揮拍層級特徵抽取：每個 unique_id 的原始 IMU 資料切成 27 次揮拍，
每次揮拍輸出 34 個特徵，存成 <unique_id>.csv（27 列 × 34 欄）。

用法：
    python src/swing_features.py --data data/39_Training_Dataset/train_data --out data/tabular_data_train
    python src/swing_features.py --data data/39_Test_Dataset/test_data --out data/tabular_data_test
"""
from pathlib import Path
import argparse
import csv
import math

import numpy as np


HEADER = [
    'ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean',
    'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var',
    'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms',
    'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min',
    'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt',
    'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy'
]


def FFT(xreal, ximag):
    """
    Cooley-Tukey radix-2 FFT（自行實作）。

    Returns:
        n: 轉換長度（不超過輸入長度的最大 2 的冪次）
        xreal: 頻譜實部
        ximag: 頻譜虛部
    """
    n = 2
    while n * 2 <= len(xreal):
        n *= 2

    xreal = xreal[:n] + [0] * (n - len(xreal))
    ximag = ximag[:n] + [0] * (n - len(ximag))

    p = int(math.log(n, 2))

    # Bit-reversal permutation
    for i in range(n):
        a = i
        b = 0
        for j in range(p):
            b = int(b * 2 + a % 2)
            a = a // 2
        if b > i:
            xreal[i], xreal[b] = xreal[b], xreal[i]
            ximag[i], ximag[b] = ximag[b], ximag[i]

    wreal = [1.0]
    wimag = [0.0]

    arg = -2 * math.pi / n
    treal = math.cos(arg)
    timag = math.sin(arg)

    for j in range(1, n // 2):
        wreal.append(wreal[-1] * treal - wimag[-1] * timag)
        wimag.append(wreal[-1] * timag + wimag[-1] * treal)

    m = 2
    while m <= n:
        for k in range(0, n, m):
            for j in range(m // 2):
                index1 = k + j
                index2 = index1 + m // 2
                t = n * j // m
                treal = wreal[t] * xreal[index2] - wimag[t] * ximag[index2]
                timag = wreal[t] * ximag[index2] + wimag[t] * xreal[index2]
                ureal = xreal[index1]
                uimag = ximag[index1]
                xreal[index1] = ureal + treal
                ximag[index1] = uimag + timag
                xreal[index2] = ureal - treal
                ximag[index2] = uimag - timag
        m *= 2

    return n, xreal, ximag


def FFT_data(input_data, swinging_times):
    """
    計算每段揮拍的加速度強度與角速度強度平均，作為 FFT 的輸入。

    Returns:
        a_mean: 加速度強度平均序列
        g_mean: 角速度強度平均序列
    """
    if not input_data or len(swinging_times) < 2 or swinging_times[-1] <= swinging_times[0]:
        return [], []

    txtlength = swinging_times[-1] - swinging_times[0]
    a_mean = [0] * txtlength
    g_mean = [0] * txtlength

    for num in range(len(swinging_times) - 1):
        a = []
        g = []
        start, end = swinging_times[num], swinging_times[num + 1]
        if start >= end or end > len(input_data):
            continue
        for swing in range(start, end):
            try:
                a.append(math.sqrt(sum(x ** 2 for x in input_data[swing][:3])))
                g.append(math.sqrt(sum(x ** 2 for x in input_data[swing][3:6])))
            except (IndexError, TypeError):
                continue
        a_mean[num] = sum(a) / len(a) if a else 0
        g_mean[num] = sum(g) / len(g) if g else 0

    return a_mean, g_mean


def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer):
    """
    對一段揮拍抽取 34 個特徵並寫成 CSV 一列。

    Args:
        input_data: 這段揮拍的資料，每列 [ax, ay, az, gx, gy, gz]
        swinging_now: 目前是第幾段揮拍
        swinging_times: 揮拍總段數
        n_fft: FFT 長度
        a_fft, g_fft: 加速度 / 角速度 FFT 實部
        a_fft_imag, g_fft_imag: 加速度 / 角速度 FFT 虛部
        writer: CSV writer
    """
    if not input_data or not a_fft or not g_fft:
        writer.writerow([0] * len(HEADER))
        return

    allsum = []
    mean = []
    var = []
    rms = []
    a = []
    g = []
    a_s1 = a_s2 = g_s1 = g_s2 = a_k1 = a_k2 = g_k1 = g_k2 = 0

    for i in range(len(input_data)):
        try:
            if i == 0:
                allsum = input_data[i]
                a.append(math.sqrt(sum(x ** 2 for x in input_data[i][:3])))
                g.append(math.sqrt(sum(x ** 2 for x in input_data[i][3:6])))
                continue
            a.append(math.sqrt(sum(x ** 2 for x in input_data[i][:3])))
            g.append(math.sqrt(sum(x ** 2 for x in input_data[i][3:6])))
            allsum = [allsum[idx] + input_data[i][idx] for idx in range(len(input_data[i]))]
        except (IndexError, TypeError):
            continue

    mean = [allsum[idx] / len(input_data) for idx in range(len(input_data[0]))] if allsum else [0] * 6
    len_data = len(input_data)

    for i in range(len_data):
        try:
            if i == 0:
                var = [(x - m) ** 2 for x, m in zip(input_data[i], mean)]
                rms = [x ** 2 for x in input_data[i]]
                continue
            var = [var[idx] + (input_data[i][idx] - mean[idx]) ** 2 for idx in range(len(input_data[i]))]
            rms = [rms[idx] + input_data[i][idx] ** 2 for idx in range(len(input_data[i]))]
        except (IndexError, TypeError):
            continue

    # 欄名是 var，實際存的是標準差
    var = [math.sqrt(v / len_data) if v > 0 else 0 for v in var] if var else [0] * 6
    rms = [math.sqrt(r / len_data) if r > 0 else 0 for r in rms] if rms else [0] * 6

    a_max = [max(a)] if a else [0]
    a_min = [min(a)] if a else [0]
    a_mean = [sum(a) / len(a)] if a else [0]
    g_max = [max(g)] if g else [0]
    g_min = [min(g)] if g else [0]
    g_mean = [sum(g) / len(g)] if g else [0]

    for i in range(len(a)):
        a_s1 += (a[i] - a_mean[0]) ** 4
        a_s2 += (a[i] - a_mean[0]) ** 2
        g_s1 += (g[i] - g_mean[0]) ** 4
        g_s2 += (g[i] - g_mean[0]) ** 2
        a_k1 += (a[i] - a_mean[0]) ** 3
        g_k1 += (g[i] - g_mean[0]) ** 3

    a_s1 = a_s1 / len_data if len_data else 0
    a_s2 = (a_s2 / len_data) ** 2 if len_data else 0
    g_s1 = g_s1 / len_data if len_data else 0
    g_s2 = (g_s2 / len_data) ** 2 if len_data else 0
    a_k2 = (a_s2 / len_data) ** 1.5 if len_data and a_s2 else 0
    g_k2 = (g_s2 / len_data) ** 1.5 if len_data and g_s2 else 0

    a_kurtosis = [a_s1 / a_s2] if a_s2 != 0 else [0]
    g_kurtosis = [g_s1 / g_s2] if g_s2 != 0 else [0]
    a_skewness = [a_k1 / a_k2] if a_k2 != 0 else [0]
    g_skewness = [g_k1 / g_k2] if g_k2 != 0 else [0]

    a_fft_mean = g_fft_mean = 0
    cut = n_fft // swinging_times if swinging_times else 1
    a_psd = []
    g_psd = []
    entropy_a = []
    entropy_g = []
    e1 = []
    e3 = []
    e2 = e4 = 0

    for i in range(cut * swinging_now, min(cut * (swinging_now + 1), len(a_fft))):
        try:
            a_fft_mean += a_fft[i]
            g_fft_mean += g_fft[i]
            a_psd.append(a_fft[i] ** 2 + a_fft_imag[i] ** 2)
            g_psd.append(g_fft[i] ** 2 + g_fft_imag[i] ** 2)
            e1.append(math.sqrt(a_psd[-1]) if a_psd[-1] >= 0 else 0)
            e3.append(math.sqrt(g_psd[-1]) if g_psd[-1] >= 0 else 0)
        except IndexError:
            continue

    a_fft_mean = a_fft_mean / cut if cut else 0
    g_fft_mean = g_fft_mean / cut if cut else 0

    a_psd_mean = sum(a_psd) / len(a_psd) if a_psd else 0
    g_psd_mean = sum(g_psd) / len(g_psd) if g_psd else 0

    for i in range(len(e1)):
        e2 += e1[i]
        e4 += e3[i]

    for i in range(len(e1)):
        try:
            if e2 > 0 and e1[i] > 0:
                entropy_a.append((e1[i] / e2) * math.log(e1[i] / e2))
            else:
                entropy_a.append(0)
            if e4 > 0 and e3[i] > 0:
                entropy_g.append((e3[i] / e4) * math.log(e3[i] / e4))
            else:
                entropy_g.append(0)
        except (ValueError, ZeroDivisionError):
            entropy_a.append(0)
            entropy_g.append(0)

    a_entropy_mean = sum(entropy_a) / len(entropy_a) if entropy_a else 0
    g_entropy_mean = sum(entropy_g) / len(entropy_g) if entropy_g else 0

    output = mean + var + rms + a_max + a_mean + a_min + g_max + g_mean + g_min + \
             [a_fft_mean, g_fft_mean, a_psd_mean, g_psd_mean] + \
             a_kurtosis + g_kurtosis + a_skewness + g_skewness + \
             [a_entropy_mean, g_entropy_mean]
    writer.writerow(output)


def generate_features(datapath, tar_dir):
    """
    掃描 datapath 下所有 .txt，切成 27 段揮拍，每個 unique_id 輸出一個特徵 CSV。
    訓練集與測試集共用這個函式，確保兩邊特徵檔格式一致。
    """
    Path(tar_dir).mkdir(parents=True, exist_ok=True)

    for file in Path(datapath).glob('**/*.txt'):
        try:
            with open(file) as f:
                All_data = []
                count = 0
                for line in f.readlines():
                    if line.strip() == '' or count == 0:
                        count += 1
                        continue
                    num = line.split()
                    if len(num) >= 6:
                        All_data.append([int(x) for x in num[:6]])

            if not All_data:
                print(f"Warning: {file} is empty or invalid, skipping.")
                continue

            swing_index = np.linspace(0, len(All_data), 28, dtype=int)

            with open(Path(tar_dir) / f'{Path(file).stem}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(HEADER)
                a_fft, g_fft = FFT_data(All_data, swing_index)
                if not a_fft or not g_fft:
                    print(f"Warning: No FFT data for {file}, writing default features.")
                    for _ in range(1, len(swing_index)):
                        writer.writerow([0] * len(HEADER))
                    continue
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                for i in range(1, len(swing_index)):
                    feature(All_data[swing_index[i - 1]:swing_index[i]], i - 1, len(swing_index) - 1,
                            n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
        except Exception as e:
            print(f"Error processing {Path(file).stem}: {type(e).__name__}: {e}")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='原始 .txt 資料夾')
    parser.add_argument('--out', required=True, help='特徵 CSV 輸出資料夾')
    args = parser.parse_args()
    generate_features(args.data, args.out)
