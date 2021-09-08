#!/bin/sh

aws dynamodb update-item \
	--table-name 'users' \
	--key '{"email": {"S": "test@test.com"}}' \
	--update-expression 'SET notifications = list_append(if_not_exists(notifications, :e), :n)' \
	--expression-attribute-values '{":n": {"L": [{"M": {"timestamp": {"N": "155634"}, "anomaly": {"S": "Test"}, "identifier": {"S": "Test"}}}]}, ":e": {"L": []}}' \
	--endpoint-url 'http://localhost:8000'



