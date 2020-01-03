# -*- coding: utf-8 -*-

import pickle
import json


class RenameUnpicklerUtil(pickle.Unpickler):
    """
    https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
    """
    def find_class(self, module, name):
        if name == 'LogisticRegressionTrain':
            from src.main.service import LogisticRegressionTrain
            return LogisticRegressionTrain
        if name == 'HierarchicalHelper':
            from src.main.algo_libs import HierarchicalHelper
            return HierarchicalHelper
        if name == 'TreeNode':
            from src.main.algo_libs import TreeNode
            return TreeNode
        if name == 'ClusterHelper':
            from src.main.algo_libs import ClusterHelper
            return ClusterHelper

        return super(RenameUnpicklerUtil, self).find_class(module, name)


def store_model(model, filename):
    fw = open(filename, 'wb')
    # 对象持久化包
    pickle.dump(model, fw)
    fw.close()


def read_model(filename):
    fr = open(filename, 'rb')
    print("load model {filename}".format(filename=filename))
    try:
        return pickle.load(fr, encoding='latin1')
    except UnicodeDecodeError:
        return pickle.load(fr)
    except ModuleNotFoundError:
        return RenameUnpicklerUtil(fr).load()
    except AttributeError:
        return RenameUnpicklerUtil(fr).load()


def check_json(input_str):
    try:
        json.loads(input_str)
        return True
    except:
        return False


def to_bool(input_str):
    return input_str.lower() in ("yes", "true", "1")
