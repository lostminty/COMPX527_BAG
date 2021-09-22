#!/usr/bin/env python3
import json
import os
import sys
from time import time
from datetime import datetime
from pprint import pp
from urllib import parse, request
from urllib.error import HTTPError

with open('output.txt', 'r', encoding='UTF-8') as file:
    DATA = file.read()

if len(sys.argv) != 4:
    print('Usage:', sys.argv[0], '<ELB url>', '<email address>', '<token>')
    sys.exit(1)

URL = f'http://{sys.argv[1]}/submit'
CHECK = {
    'email': sys.argv[2],
    'token': sys.argv[3],
    'identifier': 'Demo',
    'minimum_confidence': 10,
    'data': DATA
}

req = request.Request(URL, data=parse.urlencode(
    {'check': json.dumps(CHECK)}).encode())

print('Submission time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
start_time = time()

try:
    result = request.urlopen(req)
except HTTPError as e:
    result = e

print(f'Time taken: {(time() - start_time):.2f}')

print('Status Code:', result.getcode(), os.linesep)
result = result.read()
if result:
    parsed_result = json.loads(result)
    pp(parsed_result)
else:
    print('No JSON response.')
