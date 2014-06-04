import os
import sys



def set_paths(qn=False, data=False, out=False):
    """
    Set paths depending on machine and username
    """
    if qn:
        # qn path from run.py
        home = os.path.expanduser('~')
        path1 = os.path.join(home, 'Dropbox/nersc/csc/spikes/qn') # Cerberus
        path2 = os.path.join(home, 'csc/spikes/qn')				  # Hopper
        if not path1 in sys.path: sys.path.append(path1)
        if not path2 in sys.path: sys.path.append(path2)
        print "pathes", path1, path2
    elif out:
        # output path called by sp.py (was in sptools)
        import getpass
        path = dict()
        #hostname = socket.gethostname()
        #home = os.path.expanduser('~')
        #path['root'] = os.path.join(home, 'sn', 'py', 'spikes')
        #path['data'] = os.path.join(path['root'], 'data')            
        #path['out'] = os.path.join(path['root'], 'out')
        path['out'] = 'out'
        return path
    elif data:
        # choose data files
        home = os.path.expanduser('~')
        basedir = os.path.join(home, 'Dropbox_outsource', 'nersc', 'data') # all in one
        #basedir = os.path.join(home, 'Dropbox_outsource', 'nersc', 'data', 'EC2_CV_trials') # trials
        #basedir = os.path.join('/project/projectdirs/m636/neuro/polytrode/csc', 'data')
        files = range(1,50)
        #filenames = ['EC2_CV_trl%d.h5' % (i) for i in files] # individual trial files
        filenames = ['ecog.h5'] # all data in one file
        full = [os.path.join(basedir, f) for f in filenames]
        
        filenames = []
        for f in full:
            if os.path.exists(f): filenames.append(f)
            
        return filenames
    else:
        raise Exception("Plese specify which kind of path you are trying to get! ")

