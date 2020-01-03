import time

class HelloWordHandler(object):
    def __init__(self):
        super().__init__()

    def helloWord(self):
        time.sleep(5)
        print('Good to go!')
        return 'Good to go!'