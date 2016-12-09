# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: redis_test
# _date_ = 16/12/6 下午8:14


import redis
import json

rd = redis.Redis(host='localhost',port=6379,db=0)



doc_test= "DOCUMENT$TEST"
e_test= "EVENT$TEST"
e_train= "EVENT$TRAIN"
e_dev="EVENT$DEV"
doc_dev= "DOCUMENT$DEV"
doc_train= "DOCUMENT$TRAIN"

doc_json=rd.hget(doc_train,"src/main/resources/THYME-corpus/train/ID169_clinic_496")

print(doc_json)

js=json.loads(doc_json)
# print (js["singleWords"])

for i in js['singleWords']:
    print(i['value'])
    print(i['pre'])

print(len(js['singleWords']))
