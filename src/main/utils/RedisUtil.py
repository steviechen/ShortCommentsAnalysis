# coding=utf-8
import redis
import json
from src.main.utils import ConfigurationUtil


class RedisUtil(object):
    config_key = 'redis'

    def __init__(self):
        self.host = ConfigurationUtil.get(self.config_key, 'host')
        self.port = ConfigurationUtil.get(self.config_key, 'port')
        self.db = ConfigurationUtil.get(self.config_key, 'db')
        self.redis_pool = redis.ConnectionPool(host=self.host, port=self.port, db=self.db)
        self.redis_client = redis.Redis(connection_pool=self.redis_pool)
        # self.redis_client = redis.StrictRedis(host=self.host, port=self.port, db=self.db)

    def clear_all(self):
        """
        :return: 清空所有数据
        """
        self.redis_client.flushdb()

    def delete_key(self, keystr):
        key_list = []
        for key in self.redis_client.scan_iter(match=keystr + '*', count=10000):
            key_list.append(key)

        for key in key_list:
            self.redis_client.delete(key)

    def set_cache_data(self, contents=None, expire=3600 * 24 * 30):
        if contents is None:
            contents = {}
        for k, v in contents.items():
            self.redis_client.setex(k, expire, json.dumps(v))

    def set_single_data(self, key, value, expire=3600 * 24 * 30):
        self.redis_client.setex(key, expire, value)

    def get_cache_data(self, key):
        keys = self.redis_client.keys(key + "*")
        pipe = self.redis_client.pipeline()
        pipe.mget(keys)
        res_ls = []
        for (k, v) in zip(keys, pipe.execute()):
            nv = list(filter(lambda item: True if item is not None else False, v))
            res_ls.extend([json.loads(item) for item in nv])
        return res_ls

    def get_cache_bluk_data(self, keys):
        pipe = self.redis_client.pipeline()
        pipe.mget(keys)
        res_ls = []
        for (k, v) in zip(keys, pipe.execute()):
            nv = list(filter(lambda item: True if item is not None else False, v))
            res_ls.extend([json.loads(item) for item in nv])
        return res_ls