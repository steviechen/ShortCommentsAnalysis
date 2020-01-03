from multiprocessing import Manager

manager = Manager()
targetSplited = manager.dict()
d2vModel = manager.dict()
w2vModel = manager.dict()