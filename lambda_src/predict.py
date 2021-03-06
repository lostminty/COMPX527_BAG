import json
from time import time

import boto3

MAX_NOTIFICATIONS = 5

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('users')
sqs = boto3.resource('sqs').get_queue_by_name(QueueName='notifications')


def json_message(message):
    return json.dumps({'error': message})


def lambda_handler(event, context):
    try:
        email = event['email']
        token = event['token']
        if 'data' not in event:
            raise KeyError()
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
    except ValueError:
        return {
            'statusCode': 400,
            'body': json_message('Minimum confidence value could not be parsed.')
        }
    except:
        minimum_confidence = 20.0

    try:
        notify = bool(event['notify'])
    except ValueError:
        return {
            'statusCode': 400,
            'body': json_message('Notify boolean could not be parsed.')
        }
    except:
        notify = True

    try:
        response = table.get_item(
            Key={'email': email},
            ProjectionExpression='tokens')
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
            Payload=json.dumps(event['data']))
    except:
        return {
            'statusCode': 500,
            'body': json_message('Internal predictor error.')
        }

    payload_stream = result['Payload']
    payload = json.loads(payload_stream.read())
    payload_stream.close()

    try:
        anomaly = payload['label']
        try:
            confidence = float(payload['confidence']) * 100.0
        except:
            confidence = float('inf')
    except:
        return {
            'statusCode': 500,
            'body': json_message(f'Internal prediction result error.')
        }

    if anomaly != '-' and confidence >= minimum_confidence:
        timestamped = int(time())

        while True:
            try:
                table.update_item(
                    Key={'email': email},
                    UpdateExpression=f'REMOVE notifications[{MAX_NOTIFICATIONS - 1}]',
                    ConditionExpression='size(notifications) >= :m',
                    ExpressionAttributeValues={':m': MAX_NOTIFICATIONS})
            except:
                break

        try:
            table.update_item(
                Key={'email': email},
                UpdateExpression='SET notifications = list_append(:n, notifications)',
                ExpressionAttributeValues={
                    ':n': [{
                        'identifier': identifier,
                        'timestamp': timestamped,
                        'anomaly': anomaly
                    }]})
        except:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'anomaly': anomaly, 'confidence': confidence,
                    'error': 'Failed to store notification.'
                })
            }

        if notify:
            try:
                sqs.send_message(MessageBody=json.dumps({
                    'email': email,
                    'anomaly': anomaly,
                    'timestamp': timestamped,
                    'identifier': identifier,
                    'confidence': confidence
                }))
            except:
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'anomaly': anomaly, 'confidence': confidence,
                        'error': 'Failed to send notification.'
                    })
                }

        return {
            'statusCode': 200,
            'body': json.dumps({'anomaly': anomaly, 'confidence': confidence})
        }

    return {
        'statusCode': 200,
        'body': json.dumps({'anomaly': '-'})
    }
