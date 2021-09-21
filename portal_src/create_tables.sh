#!/bin/sh

aws dynamodb create-table \
	--table-name users \
	--attribute-definitions \
		AttributeName=email,AttributeType=S \
	--key-schema \
		AttributeName=email,KeyType=HASH \
	--provisioned-throughput \
		ReadCapacityUnits=10,WriteCapacityUnits=10 \
	--endpoint-url=http://localhost:8000

aws dynamodb create-table \
	--table-name sessions \
	--attribute-definitions \
		AttributeName=id,AttributeType=S \
	--key-schema \
		AttributeName=id,KeyType=HASH \
	--provisioned-throughput \
		ReadCapacityUnits=10,WriteCapacityUnits=10 \
	--endpoint-url=http://localhost:8000

