#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.main.utils import ConfigurationUtil
import requests
import json


class OpenTSDBUtil(object):
    config_key = 'opentsdb'

    def __init__(self, host=None, port=None):
        self.host = ConfigurationUtil.get(self.config_key, 'host') if host is None else host
        self.port = ConfigurationUtil.get(self.config_key, 'port') if port is None else port

        # 长连接写入
        self.session = requests.Session()
        self.opentsdb_save_url = "http://{host}:{port}/api/put?details".format(host=self.host, port=self.port)
        self.opentsdb_query_url = "http://{host}:{port}/api/query?".format(host=self.host, port=self.port)

    def single_insert(self, data):
        if not isinstance(data, dict):
            return
        r = self.session.post(url=self.opentsdb_save_url, json=data)
        try:
            res = json.loads(r.text)
        except:
            res = r.text
        return res

    def bulk_insert(self, data_ls, bulk_size):
        ls = []
        for item in data_ls:
            ls.append(item)
            if len(ls) == bulk_size:
                self.session.post(url=self.opentsdb_save_url, json=ls)
                ls = []
        r = self.session.post(url=self.opentsdb_save_url, json=ls)
        try:
            json.loads(r.text)
            res = {
                "rows": len(data_ls),
                "bulk_size": bulk_size
            }
        except:
            res = r.text
        return res

    def get_data_by_post(self, cond_dic):
        r = requests.post(self.opentsdb_query_url, json=cond_dic)
        print(cond_dic)
        if len(r.json()) > 0:
            print(r.json()[0]['dps'])
            dps = r.json()[0]['dps']
            return dps
        else:
            return None

    def get_data_by_post_all(self, cond_dic):
        r = requests.post("http://{host}:{port}/api/query".format(host=self.host, port=self.port), json=cond_dic)
        if len(r.json()) > 0:
            return r.json()
        else:
            return None

    def close(self):
        self.session.close()

