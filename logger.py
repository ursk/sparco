import mpi

class Logger(object):
    """
    Redirect stdout and stderr to a file and optionally echo to original stdout.
    """
    def __init__(self, stream, logFile, echo=False):
        self.out = stream
        self.logFile = logFile
        self.echo = echo

    def write(self, s):
        if len(s) == 1: return
        self.logFile.write('[%d] %s\n' % (mpi.rank, s))
        if self.echo:
            self.out.write('[%d] %s\n' % (mpi.rank, s))
