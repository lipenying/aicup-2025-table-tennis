"""
Random Forest 基礎版（Public 0.53106），保留作為版本演進對照。
原為 Google Colab 版本，路徑指向 /content/drive/MyDrive/AI/。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score


def FFT(xreal, ximag):
    """
    Perform Fast Fourier Transform on input signals.
   
    Args:
        xreal: Real part of the signal
        ximag: Imaginary part of the signal
   
    Returns:
        n: Length of the transformed signal
        xreal: Transformed real part
        ximag: Transformed imaginary part
    """
    n = 2
    while n * 2 <= len(xreal):
        n *= 2
   
    # Pad with zeros if necessary
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
    Compute FFT data for accelerometer and gyroscope signals between swing times.
   
    Args:
        input_data: List of lists containing 6 values [ax, ay, az, gx, gy, gz]
        swinging_times: List of indices marking swing boundaries
   
    Returns:
        a_mean: List of accelerometer magnitude means
        g_mean: List of gyroscope magnitude means
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
    Extract features from input data and write to CSV.
   
    Args:
        input_data: List of lists containing 6 values [ax, ay, az, gx, gy, gz]
        swinging_now: Current swing index
        swinging_times: Total number of swings
        n_fft: FFT length
        a_fft: FFT of accelerometer data
        g_fft: FFT of gyroscope data
        a_fft_imag: Imaginary part of accelerometer FFT
        g_fft_imag: Imaginary part of gyroscope FFT
        writer: CSV writer object
    """
    if not input_data or not a_fft or not g_fft:
        writer.writerow([0] * 34)
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


def data_generate():
    """
    Generate feature CSV files from training data text files.
   
    Returns:
        filename: List of file stems (unique IDs)
        all_swing: List of swing indices for each file
    """
    datapath = '/content/drive/MyDrive/AI/39_Training_Dataset/train_data'
    tar_dir = '/content/drive/MyDrive/AI/tabular_data_train'
    Path(tar_dir).mkdir(exist_ok=True)
    pathlist_txt = Path(datapath).glob('**/*.txt')
   
    filename = []
    all_swing = []
   
    for file in pathlist_txt:
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
                        tmp_list = [int(x) for x in num[:6]]
                        All_data.append(tmp_list)
               
                if not All_data:
                    print(f"Warning: {file} is empty or invalid, skipping.")
                    continue
               
                swing_index = np.linspace(0, len(All_data), 28, dtype=int)
                filename.append(int(Path(file).stem))
                all_swing.append(swing_index)
               
                headerList = [
                    'ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean',
                    'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var',
                    'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms',
                    'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min',
                    'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt',
                    'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy'
                ]
               
                with open(f'./{tar_dir}/{Path(file).stem}.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(headerList)
                    try:
                        a_fft, g_fft = FFT_data(All_data, swing_index)
                        if not a_fft or not g_fft:
                            print(f"Warning: No FFT data for {file}, writing default features.")
                            for _ in range(1, len(swing_index)):
                                writer.writerow([0] * len(headerList))
                            continue
                        a_fft_imag = [0] * len(a_fft)
                        g_fft_imag = [0] * len(g_fft)
                        n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                        n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                        for i in range(1, len(swing_index)):
                            feature(All_data[swing_index[i-1]:swing_index[i]], i - 1, len(swing_index) - 1,
                                    n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
                    except Exception as e:
                        print(f"Error processing {Path(file).stem}: {type(e).__name__}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading {Path(file).stem}: {type(e).__name__}: {e}")
            continue
   
    return filename, all_swing


def process_test_data():
    """
    Process test data and generate feature CSV files.
    """
    datapath = '/content/drive/MyDrive/AI/39_Test_Dataset/test_data'
    tar_dir = '/content/drive/MyDrive/AI/tabular_data_test'
    Path(tar_dir).mkdir(exist_ok=True)
    pathlist_txt = Path(datapath).glob('**/*.txt')
   
    for file in pathlist_txt:
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
                        tmp_list = [int(x) for x in num[:6]]
                        All_data.append(tmp_list)
               
                if not All_data:
                    print(f"Warning: {file} is empty or invalid, skipping.")
                    continue
               
                swing_index = np.linspace(0, len(All_data), 28, dtype=int)
                headerList = [
                    'ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean',
                    'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var',
                    'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms',
                    'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min',
                    'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt',
                    'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy'
                ]
               
                with open(f'./{tar_dir}/{Path(file).stem}.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(headerList)
                    try:
                        a_fft, g_fft = FFT_data(All_data, swing_index)
                        if not a_fft or not g_fft:
                            print(f"Warning: No FFT data for {file}, writing default features.")
                            for _ in range(1, len(swing_index)):
                                writer.writerow([0] * len(headerList))
                            continue
                        a_fft_imag = [0] * len(a_fft)
                        g_fft_imag = [0] * len(g_fft)
                        n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                        n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                        for i in range(1, len(swing_index)):
                            feature(All_data[swing_index[i-1]:swing_index[i]], i - 1, len(swing_index) - 1,
                                    n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
                    except Exception as e:
                        print(f"Error processing {Path(file).stem}: {type(e).__name__}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading {Path(file).stem}: {type(e).__name__}: {e}")
            continue


def main():
    """
    Main function to generate features, train models, and create submission file.
    """
    data_generate()
    process_test_data()
   
    info = pd.read_csv('/content/drive/MyDrive/AI/39_Training_Dataset/train_info.csv')
    unique_players = info['player_id'].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)
   
    datapath = '/content/drive/MyDrive/AI/tabular_data_train'
    datalist = list(Path(datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']
   
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)
   
    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            print(f"Warning: No info for unique_id {unique_id}, skipping.")
            continue
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)
        if data.empty:
            print(f"Warning: {file} is empty, skipping.")
            continue
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data), ignore_index=True)
        if player_id in train_players:
            x_train = pd.concat([x_train, data], ignore_index=True)
            y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test = pd.concat([y_test, target_repeated], ignore_index=True)
   
    scaler = MinMaxScaler()
    le = LabelEncoder()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
   
    group_size = 27


    def model_binary(X_train, y_train, X_test, y_test):
        """
        Train and evaluate a binary classification model.
       
        Args:
            X_train: Scaled training features
            y_train: Encoded training labels
            X_test: Scaled test features
            y_test: Encoded test labels
       
        Returns:
            clf: Trained classifier
        """
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
       
        predicted = clf.predict_proba(X_test)
        predicted = [predicted[i][0] for i in range(len(predicted))]
       
        num_groups = len(predicted) // group_size
        if sum(predicted[:group_size]) / group_size > 0.5:
            y_pred = [max(predicted[i*group_size:(i+1)*group_size]) for i in range(num_groups)]
        else:
            y_pred = [min(predicted[i*group_size:(i+1)*group_size]) for i in range(num_groups)]
       
        y_pred = [1 - x for x in y_pred]
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
       
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
        print(f"Binary AUC: {auc_score:.3f}")
       
        return clf


    def model_multiary(X_train, y_train, X_test, y_test):
        """
        Train and evaluate a multi-class classification model.
       
        Args:
            X_train: Scaled training features
            y_train: Encoded training labels
            X_test: Scaled test features
            y_test: Encoded test labels
       
        Returns:
            clf: Trained classifier
        """
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
       
        predicted_proba = clf.predict_proba(X_test)
        num_classes = len(clf.classes_)
        num_groups = len(predicted_proba) // group_size
       
        y_pred_proba = []
       
        for i in range(num_groups):
            group_pred = predicted_proba[i * group_size:(i + 1) * group_size]
            class_sums = [sum(group_pred[k][j] for k in range(len(group_pred))) for j in range(num_classes)]
            chosen_class = np.argmax(class_sums)
            candidate_probs = [group_pred[k][chosen_class] for k in range(len(group_pred))]
            best_instance = np.argmax(candidate_probs)
            y_pred_proba.append(group_pred[best_instance])
       
        y_test_agg = np.array([y_test[i * group_size] for i in range(num_groups)])
        y_pred_proba = np.array(y_pred_proba)
       
        present_classes = np.unique(y_test_agg)
        class_indices = [np.where(clf.classes_ == cls)[0][0] for cls in present_classes]
       
        y_pred_filtered = y_pred_proba[:, class_indices]
        y_pred_filtered = y_pred_filtered / y_pred_filtered.sum(axis=1, keepdims=True)
       
        auc_score = roc_auc_score(y_test_agg, y_pred_filtered, average='micro', multi_class='ovr', labels=present_classes)
        print(f"Multiary AUC: {auc_score:.3f}")
       
        return clf


    print("Training gender model...")
    y_train_le_gender = le.fit_transform(y_train['gender'])
    y_test_le_gender = le.transform(y_test['gender'])
    gender_model = model_binary(X_train_scaled, y_train_le_gender, X_test_scaled, y_test_le_gender)
   
    print("Training hand model...")
    y_train_le_hand = le.fit_transform(y_train['hold racket handed'])
    y_test_le_hand = le.transform(y_test['hold racket handed'])
    hand_model = model_binary(X_train_scaled, y_train_le_hand, X_test_scaled, y_test_le_hand)
   
    print("Training years model...")
    y_train_le_years = le.fit_transform(y_train['play years'])
    y_test_le_years = le.transform(y_test['play years'])
    years_model = model_multiary(X_train_scaled, y_train_le_years, X_test_scaled, y_test_le_years)
   
    print("Training level model...")
    y_train_le_level = le.fit_transform(y_train['level'])
    y_test_le_level = le.transform(y_test['level'])
    level_model = model_multiary(X_train_scaled, y_train_le_level, X_test_scaled, y_test_le_level)


    create_submission(gender_model, hand_model, years_model, level_model, scaler, le)


def create_submission(gender_model, hand_model, years_model, level_model, scaler, le):
    """
    Generate submission.csv file with model predictions.
   
    Args:
        gender_model: Trained gender classifier
        hand_model: Trained handedness classifier
        years_model: Trained play years classifier
        level_model: Trained level classifier
        scaler: MinMaxScaler object
        le: LabelEncoder object
    """
    print("Creating submission file...")
   
    test_info = pd.read_csv('/content/drive/MyDrive/AI/test_info.csv')
    test_datapath = '/content/drive/MyDrive/AI/tabular_data_test'
    test_datalist = list(Path(test_datapath).glob('**/*.csv'))
   
    submission_columns = [
        'unique_id', 'gender', 'hold racket handed',
        'play years_0', 'play years_1', 'play years_2',
        'level_2', 'level_3', 'level_4', 'level_5'
    ]
    submission_df = pd.DataFrame(columns=submission_columns)
   
    group_size = 27
   
    for file in test_datalist:
        unique_id = int(Path(file).stem)
       
        try:
            test_data = pd.read_csv(file)
            if test_data.empty:
                print(f"Warning: {file} is empty, skipping.")
                continue
           
            X_test = scaler.transform(test_data)
           
            gender_probs = gender_model.predict_proba(X_test)
            gender_pred = 1 - gender_probs[:, 0].mean()
           
            hand_probs = hand_model.predict_proba(X_test)
            hand_pred = 1 - hand_probs[:, 0].mean()
           
            years_probs = years_model.predict_proba(X_test)
            years_agg_probs = np.zeros(3)
            num_groups = max(1, len(test_data) // group_size)
            for i in range(num_groups):
                start = i * group_size
                end = min((i + 1) * group_size, len(test_data))
                group_mean = np.mean(years_probs[start:end], axis=0)
                years_agg_probs += group_mean
            years_agg_probs /= num_groups
            years_probs_dict = {f'play years_{i}': prob for i, prob in enumerate(years_agg_probs)}
           
            level_probs = level_model.predict_proba(X_test)
            num_level_classes = level_probs.shape[1]
            level_agg_probs = np.zeros(num_level_classes)
            num_groups = max(1, len(test_data) // group_size)
            for i in range(num_groups):
                start = i * group_size
                end = min((i + 1) * group_size, len(test_data))
                group_mean = np.mean(level_probs[start:end], axis=0)
                level_agg_probs += group_mean
            level_agg_probs /= num_groups
            level_probs_dict = {}
            for level in range(2, 6):
                if level in le.classes_:
                    idx = np.where(le.classes_ == str(level))[0][0]
                    level_probs_dict[f'level_{level}'] = level_agg_probs[idx]
                else:
                    level_probs_dict[f'level_{level}'] = 0.0
           
            result = {
                'unique_id': unique_id,
                'gender': gender_pred,
                'hold racket handed': hand_pred
            }
            result.update(years_probs_dict)
            result.update(level_probs_dict)
           
            submission_df = pd.concat([submission_df, pd.DataFrame([result])], ignore_index=True)
       
        except Exception as e:
            print(f"Error processing {file}: {type(e).__name__}: {e}")
            continue
   
    submission_df.to_csv('/content/drive/MyDrive/AI/submission05301.csv', index=False)
    print(f"Submission file created with {len(submission_df)} entries.")


if __name__ == '__main__':
    main()
