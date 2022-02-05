'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
import math

def make_frames(signal, hop_length, win_length):
    '''
    frames = make_frames(signal, hop_length, win_length)

    signal (num_samps) - the speech signal
    hop_length (scalar) - the hop length, in samples
    win_length (scalar) - the window length, in samples
    frames (num_frames, win_length) - array with one frame per row

    num_frames should be enough so that each sample from the signal occurs in at least one frame.
    The last frame may be zero-padded.
    '''
    frames_overlap = win_length - hop_length
    sig_length = len(signal)
    
    num_frames = np.abs(sig_length - frames_overlap) // np.abs(win_length - frames_overlap)
    rest_samples = np.abs(sig_length - frames_overlap) % np.abs(win_length - frames_overlap) 
    
    if rest_samples != 0:
        pad_sig_length = int(hop_length - rest_samples)
        z = np.zeros((pad_sig_length))
        pad_sig = np.append(signal, z)
        num_frames += 1
    else:
        pad_sig = signal
    
    idx1 = np.tile(np.arange(0, win_length), (num_frames, 1))
    idx2 = np.tile(np.arange(0, num_frames * hop_length, hop_length), (win_length, 1)).T
    
    indices = idx1 + idx2
    
    frames = pad_sig[indices.astype(np.int32, copy=False)]
#    print(np.shape(frames[0]))
    print(frames[0])
    
    return frames

def correlate(frames):
    '''
    autocor = correlate(frames)

    frames (num_frames, win_length) - array with one frame per row
    autocor (num_frames, 2*win_length-1) - each row is the autocorrelation of one frame
    '''
    mean = frames[0].mean()
    framep = frames - mean
  # autocor = [np.correlate(framep[i, :], framep[i, :], mode='full') for i in range(len(frames))]

    autocor = np.correlate(framep[0], framep[0], mode='full')
  #  print(frames[0])
 #   print(autocor)
    B = autocor.reshape(1, 479)

#    mean1 = frames[1].mean()
 #   framep1 = frames - mean    
  #  autocor1 = np.correlate(framep[1], framep[1], mode='full')
  #  C = B.reshape(1, 479)
   # B = np.append(B, [autocor1], axis=0)
    
    for i in range(1, len(frames)):
        mean = frames[i].mean()
        framep = frames - mean
        autocor = np.correlate(framep[i], framep[i], mode='full')
        B = np.append(B, [autocor], axis=0)
   #     B = autocor.reshape(1, 479) 
    return B

def make_matrices(autocor, p):
    '''
    R, gamma = make_matrices(autocor, p)

    autocor (num_frames, 2*win_length-1) - each row is symmetric autocorrelation of one frame
    p (scalar) - the desired size of the autocorrelation matrices
    R (num_frames, p, p) - p-by-p Toeplitz autocor matrix of each frame, with R[0] on main diagonal
    gamma (num_frames, p) - length-p autocor vector of each frame, R[1] through R[p]
    '''
    middle = autocor[0].size // 2
    
    p = 10
    matrix = []
    gamma = []

    for i in range(len(autocor)):   #(len(autocor)+1):
        row = []
        A = np.zeros(shape=(p,p))
       # arr = autocor[0, (middle):(middle) + p]
       # B = np.array(arr)
        for j in range(p):
        #    print(autocor[i, (middle-j):(middle-j) + p])
            arr = autocor[i, (middle-j):(middle-j) + p]
            A[j] = arr

        matrix.append(A)
        gam1 = autocor[i, (middle):(middle) + p + 1]
        gamma.append(gam1[1:p+1])
    matrix = np.array(matrix)
    gamma = np.array(gamma)
   # print(gamma)
  #  print(len(gamma[0]))
  #  print(matrix[0, :6, :6])
    return matrix, gamma

def lpc(R, gamma):
    '''
    a = lpc(R, gamma)
    Calculate the LPC coefficients in each frame

    R (num_frames, p, p) - p-by-p Toeplitz autocor matrix of each frame, with R[0] on main diagonal
    gamma (num_frames, p) - length-p autocor vector of each frame, R[1] through R[p]
    a (num_frames,p) - LPC predictor coefficients in each frame
    '''
    a = []
 #   print(len(R))
    for i in range(6):
        R_inv = np.linalg.inv(R[i])
        product = np.dot(R_inv, gamma[i])
        a.append(product)

    arr = np.array(a)
    return arr

def framepitch(autocor, Fs):
    '''
    framepitch = framepitch(autocor, samplerate)

    autocor (num_frames, 2*win_length-1) - autocorrelation of each frame
    Fs (scalar) - sampling frequency
    framepitch (num_frames) - estimated pitch period, in samples, for each frame, or 0 if unvoiced

    framepitch[t] = 0 if the t'th frame is unvoiced
    framepitch[t] = pitch period, in samples, if the t'th frame is voiced.
    Pitch period should maximize R[framepitch]/R[0], in the range 4ms <= framepitch < 13ms.
    Call the frame voiced if and only if R[framepitch]/R[0] >= 0.3, else unvoiced.
    '''
    periods = []   
    for frame in range(len(autocor)):
        threshold = {}
        R_0 = (autocor[frame,239]) 
        for i in range(32,105):
   #     print(autocor[frame, (239+i+1)] / (autocor[frame, 239]))
            if ((autocor[frame,(239+i)])/ autocor[frame,239] >= 0.3):
                threshold[(autocor[frame, (239+i)] / (autocor[frame, 239]))] = i
        if len(threshold) == 0:
            periods.append(0)
    #    else:
    #print(max(threshold))
        else:
            max_t = max(threshold)
            periods.append(threshold.get(max_t))
            
    print(len(periods))
    return periods
            
def framelevel(frames):
    '''
    framelevel = framelevel(frames)

    frames (num_frames, win_length) - array with one frame per row
    framelevel (num_frames) - framelevel[t] = power (energy/duration) of the t'th frame, in decibels
    '''
    [num_frames, win_length] = frames.shape
    framelevel = np.zeros((num_frames))
    for i in range(num_frames):
        power = np.sum(frames[i,:]**2 / win_length)
        framelevel[i] = (10 * math.log10(power))   
        
   # rounded = np.round(framelevel)
    return framelevel

def interpolate(framelevel, framepitch, hop_length):
    '''
    samplelevel, samplepitch = interpolate(framelevel, framepitch, hop_length)

    framelevel (num_frames) - levels[t] = power (energy/duration) of the t'th frame, in decibels
    framepitch (num_frames) - estimated pitch period, in samples, for each frame, or 0 if unvoiced
    hop_length  (scalar) - number of samples between start of each frame
    samplelevel ((num_frames-1)*hop_length+1) - linear interpolation of framelevel
    samplepitch ((num_frames-1)*hop_length+1) - modified linear interpolation of framepitch

    samplelevel is exactly as given by numpy.interp.
    samplepitch is modified so that samplepitch[n]=0 if the current frame or next frame are unvoiced.
    '''
    xp = []
    x = []

    for i in range(0, len(framelevel)):
        xp.append(i*hop_length)

    for j in range((len(framelevel)-1)*hop_length+1):
        x.append(j)
        
    num_frames = len(framelevel)

    samplelevel = np.interp(x, xp, framelevel)


    samplepitch = []
    xp1 = []
    x1 = []    

  #  framepitch2 = framepitch

    for i in range(0, len(framelevel)):
        xp1.append(i*hop_length)
    for j in range((len(framelevel)-1)*hop_length+1):
        x1.append(j)
      
    samplepitch = np.interp(x1, xp1, framepitch)
    
    for frame in xp:
        if (samplepitch[frame] == 0 or (frame+1 < num_frames) and samplepitch[frame+1] == 0):
            samplepitch[frame: frame+hop_length] = 0
            samplepitch[frame-hop_length:frame] = 0
    print(samplelevel[0:10])
  
    return samplelevel, samplepitch

def excitation(samplelevel, samplepitch):
    '''
    phase, excitation = excitation(samplelevel, samplepitch)

    samplelevel ((num_frames-1)*hop_length+1) - effective level (in dB) of every output sample
    samplepitch ((num_frames-1)*hop_length+1) - effective pitch period for every output sample
    phase ((num_frames-1)*hop_length+1) - phase of the fundamental at every output sample,
      modulo 2pi, so that 0 <= phase[n] < 2*np.pi for every n.
    excitation ((num_frames-1)*hop_length+1) - LPC excitation signal
      if samplepitch[n]==0, then excitation[n] is zero-mean Gaussian
      if samplepitch[n]!=0, then excitation[n] is a delta function time-aligned to the phase
      In either case, excitation is scaled so its average power matches samplelevel[n].
    '''
    ######## WARNING: the following lines must remain, so that your random numbers will match the grader
    from numpy.random import Generator, PCG64
    rg = Generator(PCG64(1234))
    ## Your Gaussian random numbers must be generated using the command ***rg.normal***
    ## (See https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html).
    ## (1) You must generate them in order, from the beginning to the end of the waveform.
    ## (2) You must generate a random sample _only_ if the corresponding samplepitch[n] > 0.
    
    phase = np.zeros((len(samplepitch)))
    excitation = np.zeros((len(samplepitch)))
    for i in range(len(phase)):
        arms = np.sqrt(10**(samplelevel[i]/10))
        if samplepitch[i] > 0:
            arms = np.sqrt(10**(samplelevel[i]/10))
            phase[i] = (phase[i-1] + (2*np.pi/samplepitch[i])) % (2*np.pi)
            if phase[i-1] > phase[i]:
                excitation[i] = np.sqrt(samplepitch[i])*arms    
        else:
            phase[i] = phase[i-1]
            excitation[i] = rg.normal(0)*arms
                
    #print(samplepitch[6000:6500])
  #  print(phase[15000:15500])
    #for sample in range(len(samplepitch)):
        #arms = np.sqrt(10**(sample/10))
        #adelta = arms*np.sqrt(1/samplepitch[sample])
     #   if samplepitch[sample] == 0:
      #      excitation[sample] = 0
       # else:
        #    excitation[sample] = rg.normal(samplepitch[sample])

    #print(excitation[0:10])

    return phase, excitation

def synthesize(excitation, a):
    '''
    y = synthesize(excitation, a)
    excitation ((num_frames-1)*hop_length+1) - LPC excitation signal
    a (num_frames,p) - LPC predictor coefficients in each frame
    y ((num_frames-1)*hop_length+1) - LPC synthesized  speech signal
    '''
    num_frames = a.shape[0]
    hop_length = (excitation.shape[0] - 1)/ (num_frames - 1)
    p = a.shape[1]
    y = np.zeros(excitation.shape[0])
    
    
    for i in range(0, excitation.shape[0]):
        frame = int(np.floor(i/hop_length))
        hold = 0
        for j in range(0,p):
            if frame>0 or i>=j:
                hold += a[frame][j]*y[i-j - 1]
        y[i] = excitation[i] + hold
        #y = excitation[i] + a*(y-i)
    
    return y
    #print(len(excitation))
