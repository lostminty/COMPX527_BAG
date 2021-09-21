import json
from time import time

import boto3

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('users')
sqs = boto3.resource('sqs').get_queue_by_name(QueueName='notifications')


def json_message(message):
    return json.dumps({'error': message})


def lambda_handler(event, context):
    try:
        email = event['email']
        token = event['token']
        image = event['image']
    except:
        return {
            'statusCode': 400,
            'body': json_message('Missing required fields.')
        }

    try:
        identifier = event['identifier']
        if len(identifier) > 32:
            return {
                'statusCode': 400,
                'body': json_message('Identifier is too long.')
            }
    except:
        identifier = '-'

    try:
        minimum_confidence = float(event['minimum_confidence'])
    except:
        minimum_confidence = 50.0

    try:
        response = table.get_item(Key={'email': email})
        token_present = token in response['Item']['tokens']
    except:
        token_present = False

    if not token_present:
        return {
            'statusCode': 401,
            'body': json_message('Bad token.')
        }

    try:  # This is where we get our classification.
        result = boto3.client('lambda').invoke(
            FunctionName='predictor',
            Payload=json.dumps({'image': image}))
        payload = json.loads(result['Payload'])
        anomaly = payload['anomaly']
        confidence = payload['confidence']
    except:
        return {
            'statusCode': 500,
            'body': json_message('Internal error.')
        }

    if anomaly and confidence >= minimum_confidence:
        timestamped = int(time())

        sqs.send_message(MessageBody=json.dumps({
            'email': email,
            'anomaly': anomaly,
            'timestamp': timestamped,
            'identifier': identifier,
            'confidence': confidence
        }))

        table.update_item(
            Key={'email': email},
            UpdateExpression='SET notifications = list_append(notifications, :n)',
            ExpressionAttributeValues={
                ':n': [
                    {
                        'identifier': identifier,
                        'timestamp': timestamped,
                        'anomaly': anomaly
                    }
                ],
            })

    return {
        'statusCode': 200,
        'body': json.dumps({'anomaly': anomaly})
    }
