class Logger:
    def __init__(self, filename):
        self.filename = filename + '.txt'
    def write(self, *argv):
        argv = [str(arg) for arg in argv]
        print('logger write', ' '.join(argv))
        with open(self.filename,'a') as f:
            f.write(' '.join(argv) +'\n')