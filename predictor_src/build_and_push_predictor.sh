#!/bin/bash
set -e

name=predictor

if [ -z "$AWS_ID" ]; then
	AWS_ID=$(aws sts get-caller-identity --query Account --output text)
fi

url=$AWS_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# In case this script is called from elsewhere, we must always build relative
# to where this script is.
docker build -t $name "$(cd -- "$(dirname "$0")" > /dev/null 2>&1; pwd -P)"

aws ecr get-login-password | docker login \
	--username AWS --password-stdin $url

aws ecr describe-repositories --repository-names $name || \
	aws ecr create-repository \
		--repository-name $name \
		--image-scanning-configuration scanOnPush=true \
		--image-tag-mutability MUTABLE

docker tag $name:latest "$url/$name:latest"
docker push "$url/$name:latest"

docker logout $url

if aws lambda list-functions --query 'Functions[*].[FunctionName]' \
	--output text | grep -q $name
then
	aws lambda update-function-code \
		--function-name $name \
		--image-uri $url/$name:latest
else
	aws lambda create-function \
		--function-name $name \
		--role "arn:aws:iam::$AWS_ID:role/${name^}" \
		--package-type Image \
		--timeout 600 \
		--memory-size 1024 \
		--code ImageUri=$url/$name:latest
fi
