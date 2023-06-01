import librosa
import numpy as np
import fathon
from fathon import fathonUtils as fu
from scipy import stats
from scipy import signal as sig


def resampling(audio, orig_sr, new_sr):
    """Resample audio to new sampling rate.
    Args:
        audio: original audio signal to resample.
        orig_sr: original sampling rate.
        new_sr: new sampling rate.
    Returns:
        signal: signal with new sampling rate.
    """
    signal = librosa.resample(audio, orig_sr=orig_sr, target_sr=new_sr)
    return signal


def featureSelector(
    signal, sr, feature, pre_emph=False, pos_norm="default", cwt_width=32, **kwargs
):
    """Extract selected feature from input audio signal.
    Args:
        x: input audio signal.
        sr: sampling rate of audio.
        feature: feature to extract.
        **kwargs:
            hop_length: Number of samples between successive frames.
            n_fft: Length of the FFT window.
            win_size: Size of moving window.
            n_mfcc: Number of MFCCs to return.
            ...
    Returns:
        extracted: Extracted feature.
    """
    sp_feature = [
        "ste",
        "zcr",
        "acr",
        "centroid",
        "rolloff",
        "bandwidth",
        "mfcc",
        "dfa",
        "melspec",
        "tempogram",
        "ma",
        "msd",
        "cwt",
    ]
    assert feature.split("_")[0] in sp_feature, "ERROR IN FEATURE SPECIFIED"
    if pre_emph:
        signal = librosa.effects.preemphasis(signal, coef=0.97)
    norm_audio = pre_normalize(signal)
    if "ste" in feature:
        extracted = librosa.feature.rms(y=norm_audio, **kwargs)
    elif "zcr" in feature:
        extracted = librosa.feature.zero_crossing_rate(y=norm_audio, **kwargs)
    elif "acr" in feature:
        extracted = librosa.feature.tempogram(y=norm_audio, sr=sr, **kwargs)
    elif "centroid" in feature:
        extracted = librosa.feature.spectral_centroid(y=norm_audio, sr=sr, **kwargs)
    elif "rolloff" in feature:
        extracted = librosa.feature.spectral_rolloff(y=norm_audio, sr=sr, **kwargs)
    elif "bandwidth" in feature:
        extracted = librosa.feature.spectral_bandwidth(y=norm_audio, sr=sr, **kwargs)
    elif "mfcc" in feature:
        extracted = librosa.feature.mfcc(y=norm_audio, sr=sr, **kwargs)
    elif "dfa" in feature:
        y = fu.toAggregated(norm_audio)
        dfa_obj = fathon.DFA(y)
        window = fu.linRangeByCount(0, len(norm_audio), **kwargs)
        _, extracted = dfa_obj.computeFlucVec(window, revSeg=True, polOrd=3)
    elif "melspec" in feature:
        extracted = librosa.feature.melspectrogram(y=norm_audio, sr=sr, **kwargs)
    elif "tempogram" in feature:
        extracted = librosa.feature.tempogram(y=norm_audio, sr=sr, **kwargs)
    elif "ma" in feature:
        extracted, _ = moving_stats(x=norm_audio, **kwargs)
    elif "msd" in feature:
        _, extracted = moving_stats(x=norm_audio, **kwargs)
    elif "cwt" in feature:
        widths = np.arange(1, cwt_width)
        extracted = np.array([sig.cwt(norm_audio[0], sig.ricker, widths)])
    else:
        print("Error in feature")
        return
    norm_extracted = post_normalize(extracted, pos_norm)
    return norm_extracted


def MFCC(audio, sampling_rate, pos_norm="default", **kwargs):
    """Perform MFCC feature extraction
    Args:
        audio: input audio signal.
        sampling_rate: sampling rate of audio.
        pos_norm: method for post normalization.
        **kwargs:
            hop_length: Number of samples between successive frames.
            n_fft: Length of the FFT window.
            win_size: Size of moving window.
            n_mfcc: Number of MFCCs to return.
            ...
    Returns:
        normalized mfcc feature map.
    """
    norm_audio = pre_normalize(audio)
    mfcc = librosa.feature.mfcc(y=norm_audio, sr=sampling_rate, **kwargs)
    norm_mfcc = post_normalize(mfcc, pos_norm)
    return norm_mfcc


def pre_normalize(audio, range=1, mean=0):
    """Pre normalization.
    Args:
        audio: input audio signal.
        range: the traget amplitude of waveform.
        mean: the target mean value of waveform.
    Returns:
        normalized audio signal
    """
    maximum = np.max(audio)
    minimum = np.min(audio)
    bias = (range - mean) / 2
    if maximum == minimum:
        return 0
    else:
        return (audio - minimum) * range / (maximum - minimum) - bias


def post_normalize(feature, method="zscore"):
    """Post normalization.
    Args:
        feature: input MFCC feature map.
        method: normaliztion method (default is librosa.util.normalize).
    Returns:
        normalized feature map.
    """
    if method == "zscore":
        return stats.zscore(feature, axis=None)
    elif method == "cmvn":
        cepstral_mean = np.mean(feature, axis=1, keepdims=True)
        feature_centered = feature - cepstral_mean
        cepstral_variance = np.var(feature_centered, axis=1, keepdims=True)
        return feature_centered / np.sqrt(cepstral_variance)
    elif method == "default":
        return librosa.util.normalize(feature)
    else:
        return feature


def moving_stats(x, win_length, hop_length):
    """Obtain moving statistics of an audio signal based on framing anaysis.
    Args:
        x: input audio signal.
        win_lenght: window length for moving window.
        hop_length = Number of saples between successie frames.
    Returns:
        ma, msd: Moving average and moving standard
        statistics of the input signal.
    """
    i = 0
    ma = []
    msd = []
    for i in range(0, len(x) - hop_length, hop_length):
        avg = round(np.sum(x[i : i + win_length]) / win_length, 2)
        sd = round(np.std(x[i : i + win_length]), 2)
        ma.append(avg)
        msd.append(sd)
    return ma, msd


def butterworth_coefficient(f_low, f_high, sampling_rate, order=5, ftype="band"):
    """Generates coefficients for butterworth filter.
    Args:
        f_low: Lowest cut-off frequency.
        f_high: Highest cut-off frequency.
        sampling_rate: Sampling rate for nyquist frequency calculation
        order:  Order of filter.
        ftype: Type of filtering, default as band pass :bandpass, lowpass and highpass.
    Returns:
        b, a coeffiicents of butterworth filter generated from specification.
    """
    nyq = 0.5 * sampling_rate
    low = f_low / nyq
    high = f_high / nyq
    b, a = sig.butter(order, [low, high], btype=ftype)
    return b, a


def butterworth_bandpass_filter(x, f_low, f_high, sampling_rate, order=5):
    """Filter input audio based on butterworth coefficients obtained for].
    Args:
        x: Input signal.
        f_low: Lowest cut-off frequency.
        f_high: Highest cut-off frequency.
        sampling_rate: Sampling rate for nyquist frequency calculation
        order:  Order of filter.
    Returns;
        filtered audio: The output of the butterworth filter.
    """
    b, a = butterworth_coefficient(f_low, f_high, sampling_rate, order=order)
    filtered_audio = sig.lfilter(b, a, x)
    return filtered_audio


def feature_stft(x, sampling_rate, window_type="hann"):
    """Obtain STFT features from an audio signal.
    Args:
        x: input audio signal.
        sampling_rate: sampling frequency of audio
        hop_length = Number of saples between successie frames.
        window_type = Length of moving window.
    Returns:
        STFT: STFT of the input signal.
    """
    fs = sampling_rate
    audio = pre_normalize(x)
    audio = butterworth_bandpass_filter(audio, 1, 4000, fs, order=3)
    stft = librosa.stft(
        audio, n_fft=int(0.02 * fs), hop_length=int(0.01 * fs), window=window_type
    )
    stft = librosa.amplitude_to_db(stft[0 : int(len(stft) / 2), :], ref=np.max)
    return stft


def feat_corr_coeff2D(A, B):
    """Obtain correlation coefficients between two 2D variables.
    (Only supports Pearson's Correlation currently.)
    Args:
        A: 2D variable in the form of list or np array.
        B: second 2D variable in the form of list or np array.
    Returns:
        coef: Pearson's correlation coefficient between the two variables.
    """
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    coef = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))
    # Finally get corr coeff
    return coef


def feat_corr_coeff1D_2D(A, B):
    """Obtain correlation coefficients between two variables of different dim.
    (Only supports Pearson's Correlation currently.)
    Args:
        A: 1D variable in the form of list or np array.
        B: 2D variable in the form of list or np array.
    Returns:
        coef: Pearson's correlation coefficient between the two variables.
    """
    A_mA = np.sum(A - np.mean(A))
    B_mB = np.sum(B - np.mean(B, axis=1))[:, None]

    ssA = np.sum((A_mA**2))
    ssB = np.sum((B_mB**2))[None]
    coef = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

    return coef
