import thinkdsp
from scipy.linalg import toeplitz
from scipy.fftpack import fft,ifft
import multiprocessing
import functools
import musdb
import librosa
import librosa.display
import math



'''
mus = musdb.DB(
    root="/home/nnanos/open-unmix-pytorch/musdb18_wav",
    is_wav=True,
    subsets='test',
)
a = mus.tracks
tracks = a[0]
tmp = thinkdsp.np.reshape(tracks.stems , tracks.stems.shape)


#normalizing the sound only silences the sound (normalizing its energy to 1)
thinkdsp.sf.write('/home/nnanos/Desktop/test_sound.wav', tmp[0,:,0].astype("float64") , 44100)
thinkdsp.sf.write('/home/nnanos/Desktop/filtered_sound.wav', tmp[0,:,0]/(thinkdsp.np.dot(tmp[0,:,0],tmp[0,:,0])).astype("float64") , 44100)
thinkdsp.sf.write('/home/nnanos/Desktop/test_sound.wav', (tmp[4,:,0]+tmp[1,:,0]+tmp[2,:,0]+tmp[3,:,0]).astype("float64") , 44100)


#check if sources are orthogonal:
thinkdsp.np.dot(tmp[1,:,0],tmp[3,:,0])
thinkdsp.np.dot(tmp[1,:,0]/(thinkdsp.np.dot(tmp[1,:,0],tmp[1,:,0])),tmp[4,:,0]/(thinkdsp.np.dot(tmp[4,:,0],tmp[4,:,0])))

'''




'''
N_samples = len(x)//30
x = x[:N_samples]

#convolution using Toeplitz matrix

h_padded = thinkdsp.np.pad(h , (0,N_samples-(len(h))), 'constant')

e1 = thinkdsp.np.zeros(N_samples)
e1[0] = 1

T = toeplitz( e1/16 , h_padded )

y = T.dot(x)


#convolution using DFT
y = ifft( fft(x)*( fft(h,N_samples) ) )

#downsampling stage
tmp1 = thinkdsp.np.eye(N_samples//2)
tmp2 = thinkdsp.np.zeros((N_samples//2,N_samples//2))

D_N = thinkdsp.np.concatenate((tmp1,tmp2) , axis=0)

e2 = thinkdsp.np.array([1,0])
x_filteredAndSubsampled = thinkdsp.np.kron(D_N,e2)
'''


def get_one_level_chunk(sig_chunk):
    #h = [1 , 4 , 6 , 4 , 1]
    N_samples = len(sig_chunk)

    
    #convolution using DFT (low pass filtering with gaussian kernel)
    sig_chunk_y = ifft( fft(sig_chunk)*( fft(h,N_samples) ) )
    sig_chunk_y = thinkdsp.np.real(sig_chunk_y)
    

    '''
    #convolution using Toeplitz matrix

    h_padded = thinkdsp.np.pad(h , (0,N_samples-(len(h))), 'constant')

    e1 = thinkdsp.np.zeros(N_samples)
    e1[0] = 1

    T = toeplitz( e1/16 , h_padded )
    '''

    #downsampling stage----------------------------------------------------------
    tmp1 = thinkdsp.np.eye(N_samples//2)
    tmp2 = thinkdsp.np.zeros((N_samples//2,N_samples//2))

    D_N = thinkdsp.np.concatenate((tmp1,tmp2) , axis=0)
    e2 = thinkdsp.np.array([1,0])
    Downsampling_matrix = thinkdsp.np.kron(D_N,e2)
    sig_chunk_y = sig_chunk_y[:Downsampling_matrix.shape[1]]

    x_chunk_filteredAndSubsampled = Downsampling_matrix.dot(sig_chunk_y)
    x_chunk_filteredAndSubsampled = x_chunk_filteredAndSubsampled[:N_samples//2]
    #------------------------------------------------------------------------------

    x_chunk_filtered = sig_chunk_y

    return x_chunk_filteredAndSubsampled , x_chunk_filtered


def get_one_level_total(signal , samplerate , h):
    x_filteredAndSubsampled_list_of_chunks = []

    #if the mod =! 0 then 
    if  (len(signal) % samplerate) :
        nb_secs = math.ceil(len(signal)/samplerate)
        nb_samples = nb_secs*samplerate 
        tmp = thinkdsp.np.zeros((nb_samples,))
        tmp[:len(signal)] = signal
        signal = tmp


    sig_chunks = thinkdsp.np.split(signal , len(signal)//samplerate )

    '''
    #without multiprocessing
    for chunk in sig_chunks:
        chunk_x_filtered_downsampled = get_one_level_chunk(chunk)
        x_filteredAndSubsampled_list_of_chunks.append(chunk_x_filtered_downsampled)
    '''
    
    #with multiprocessing
    pool = multiprocessing.Pool(4)

    list_of_chunks_filteredAndSubsampled_AndFilteredOnly = pool.map(
        get_one_level_chunk,
        iterable=sig_chunks,
        chunksize=10
    )
        
    pool.close()
    pool.join()

    x_filteredAndSubsampled_list_of_chunks = []
    x_filtered_list_of_chunks = []
    for chunk_filteredAndSubsampled , chunk_filtered in list_of_chunks_filteredAndSubsampled_AndFilteredOnly:
        x_filtered_list_of_chunks.append(chunk_filtered)
        x_filteredAndSubsampled_list_of_chunks.append(chunk_filteredAndSubsampled) 



    arr = thinkdsp.np.array(x_filteredAndSubsampled_list_of_chunks)
    lvl_output = arr.flatten()

    arr = thinkdsp.np.array(x_filtered_list_of_chunks)
    lvl_gauss_pyr = arr.flatten()


    return lvl_output , lvl_gauss_pyr ,signal


#test-----------------------------------------------------------------------------------------------------

#first level
x , s = thinkdsp.librosa.load("/home/nnanos/Downloads/ThinkDSP-master-20200928T154642Z-001/ThinkDSP-master/code/Madvillain - Supervillain Theme - Madvillainy (Full Album).wav",sr=11025)

#second level
#x,s = thinkdsp.librosa.load("/home/nnanos/Desktop/filtered_signal2.wav",sr=s//2)

h = [ 4 , 6 , 4 ]
signal_out_filtered_and_subsampled , filtered_signal_gauss_pyramid , x = get_one_level_total(x,s,h)
thinkdsp.sf.write('/home/nnanos/Desktop/filtered_signal2.wav', thinkdsp.np.abs((x[:len(filtered_signal_gauss_pyramid)]-filtered_signal_gauss_pyramid)).astype("float64") , s)

'''
#seeing the spectrogram
X = librosa.stft(filtered_signal_gauss_pyramid.T , n_fft=4096)
librosa.display.specshow(librosa.amplitude_to_db(thinkdsp.np.abs(X), sr=s , ref=thinkdsp.np.max), y_axis='log', x_axis='time')
thinkdsp.plt.show()
'''

a = 0

