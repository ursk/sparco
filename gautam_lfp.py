""" 
some tools to map LFP coefficients on the linear track and make sense of it

to create these plots:
1. preprocessing data with the ../data/gautam_preprocess.m script
2. learn an LFP basis with run.py
3. sparsify the data with prallel_sparsify.py 

"""

#[done] plot ground truth position as a function of position to debug.

## this does not work
#import scipy.io
#scipy.io.loadmat('/Users/urs/Dropbox/nersc/data/gautam_ec014-468Dec4.mat')

# awesome bit of code that loads up all variables from a mat or hd5 file. 
import h5py
def h5all(h5name):
    h5 = h5py.File(h5name, 'r')
    for key in h5.keys():
    			print key
    			globals()[key] = h5[key][:]
    h5.close()

h5name = '/Users/urs/Dropbox/nersc/data/gautam_ec014-468Dec4.mat'
h5all(h5name) # raw data
h5name = '/Users/urs/Dropbox/nersc/data/0066_micro_climate_coefficientstest_merge.h5'
h5all(h5name) # sparse coefficients




# now to the unwrapping that Gautam was talking about: 
# X    - 425340 raw datas
# tr   - the trial number
# fast - if the rat is moving fast
# p1   - the place 
# data - 212733x100 downsampled sparse components

placebin = np.int64(np.round(50*(p1.flatten()[::2]+.14))) # turn p1 position information into place (binned) as function of time
trialpos = np.int64(tr.flatten()[::2]) 

for basis in range(10**2):
    all_row=np.zeros((np.max(placebin)+1, np.max(trialpos))) # 114 place bins, 89 trials
    for trial in range(np.max(trialpos)):
        inx = find(trialpos==trial+1)
        all_row[:, trial] =  np.bincount(placebin[inx], weights=data[inx,basis], minlength=np.max(placebin)+1)
        all_row[:, trial] /= np.bincount(placebin[inx],                          minlength=np.max(placebin)+1) # normalization 
    plt.subplot(10,10,basis+1)
    all_row[np.isnan(all_row)==1]=0 # make NaN white
    plt.imshow(-all_row.T, vmin=-1, vmax=0)
    plt.xticks([]); plt.yticks([])
    plt.gray()
    print ".",






# debug with ground truth -- now works! 
fakedata = p1.flatten()[::2] # position(time)
all_row=np.zeros((np.max(placebin)+1, np.max(trialpos)))
for trial in range(np.max(trialpos)):
    inx = find(trialpos==trial+1) # time points for this trial 
    all_row[:, trial] = np.bincount(placebin[inx], weights=fakedata[inx], minlength=np.max(placebin)+1) # placebin is a discretized version of position p1
    all_row[:, trial] /= np.bincount(placebin[inx],  minlength=np.max(placebin)+1)
plt.subplot(121); plt.imshow(all_row.T, interpolation='nearest'); plt.title('position')

fakedata = trialpos # trialnumber(time)
all_row=np.zeros((np.max(placebin)+1, np.max(trialpos)))
for trial in range(np.max(trialpos)):
    inx = find(trialpos==trial+1)
    all_row[:, trial] = np.bincount(placebin[inx], weights=fakedata[inx], minlength=np.max(placebin)+1)
    all_row[:, trial] /= np.bincount(placebin[inx],  minlength=np.max(placebin)+1)
plt.subplot(122); plt.imshow(all_row.T, interpolation='nearest'); plt.title('trial number')

# target: aim for
plot(p1.flatten()) # sweep through all positions in each trial.  
plot(np.diff(tr.flatten())) # these line up well, but after resampling,  

plot(p1.flatten()[tr.flatten()==71]) # looks terrible. 



    