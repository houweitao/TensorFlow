# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: json_decode
# _date_ = 16/12/6 下午8:04

import json

html = "{\"msg\":\"invalid_request_scheme: http\",\"code\":100,\"request\":\"GET \/v2\/book\/isbn\/9787218087351\"}"

hjson = json.loads(html)

print(hjson)

print (hjson['msg'])
# print (hjson['images']['large'])
print (hjson['code'])
