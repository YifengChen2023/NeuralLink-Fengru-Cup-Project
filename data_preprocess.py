import os
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from scipy import signal
import numpy as np
# from numpy.fft import fft, fftfreq
import mne
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
from mne.viz import plot_filter, plot_ideal_filter
# from mne_connectivity import envelope_correlation


def dataPreprocess(raw_data):

    # print('raw_data.shape: ' + str(raw_data.shape))


    # ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8','AF4']
    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'FC6', 'F4', 'F8', 'AF4']
    # ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']

    sfreq = 128

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

    # 展示数据信息
    # print(raw)
    # print(raw.info)
    # raw.plot_sensors(ch_type='eeg', show_names=True)
    # raw.plot(duration=5, n_channels=len(raw.ch_names), scalings=dict(eeg=100e-6), clipping=None)
    # raw.plot_psd(average=True)
    # print('raw_data: ' + str(raw.get_data()))

    # FIR滤波
    # filter_params = mne.filter.create_filter(raw.get_data(), raw.info['sfreq'], l_freq=8, h_freq=30, fir_design='firwin')
    # mne.viz.plot_filter(filter_params, raw.info['sfreq'], flim=(0.1, 64))

    # raw_bandpass = raw.copy().filter(l_freq=8, h_freq=30, method='fir')
    # print('raw_bandpass_data: ' + str(raw_bandpass.get_data()))
    # print(raw_bandpass.get_data()[0])
    # with mne.viz.use_browser_backend('matplotlib'):
    #     fig = raw_bandpass.plot(duration=5, n_channels=len(raw.ch_names), scalings=dict(eeg=100e-6), clipping=None)
    #     fig.suptitle('Band-pass filtered at 8-30 Hz')
    # raw_bandpass.plot_psd(average=True)

    # 降采样
    raw_downsampled = raw.copy().resample(sfreq=88)
    # plt.show()

    return raw_downsampled.get_data()
    # ICA
    # ica = ICA(max_iter='auto')
    # ica.fit(raw_bandpass)
    # ica.plot_sources(raw_bandpass)
    # ica.plot_components()

    # 数据分段
    epochs = mne.make_fixed_length_epochs(raw_bandpass, duration=1, preload=False)
    # epochs.plot(scalings=dict(eeg=100e-6))
    # epochs.plot_psd(picks='eeg')

    # 数据叠加平均
    evoked = epochs.average()
    # 绘制逐导联的时序信号图
    # evoked.plot()
    # 绘制逐导联热力图
    # evoked.plot_image()
    # 绘制平均所有电极后的ERP
    # mne.viz.plot_compare_evokeds(evokeds=evoked, combine='mean')

    # 时频分析
    freqs = np.logspace(*np.log10([8, 30]), num=10)
    n_cycles = freqs / 2
    power = tfr_morlet(evoked, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False)
    # axe1 = plt.axes()
    # axe1.xaxis.set_visible(False)
    # axe1.yaxis.set_visible(False)

    # figs = power.plot(title='auto', colorbar=False, show=True)

    # figs[0].savefig('waveletTransform/train/rh/rh' + str(fig_index)  + '.png', bbox_inches='tight', pad_inches=0, dpi=30)
    # figs[1].savefig('waveletTransform/train/rh/rh' + str(fig_index)  + '1.png', bbox_inches='tight', pad_inches=0, dpi=30)
    # fig.savefig('waveletTransform/train/rh/rh' + str(fig_index)  + '.png', bbox_inches='tight', pad_inches=0, dpi=30)
    # img = cv2.imread('waveletTransform/train/rh/rh' + str(fig_index)  + '.png', cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite('waveletTransform/train/rh/rh' + str(fig_index)  + '.png', img)
    # plt.show()