#!/bin/sh

aws dynamodb update-item \
	--table-name 'users' \
	--key '{"email": {"S": "compx527brms@gmail.com"}}' \
	--update-expression 'SET notifications = list_append(if_not_exists(notifications, :e), :n)' \
	--expression-attribute-values '{":n": {"L": [{"M": {"timestamp": {"N": "155634"}, "anomaly": {"S": "Test"}, "identifier": {"S": "<h1>Injection Test</h1>"}}}]}, ":e": {"L": []}}' 



