import json
from datetime import datetime

import boto3

SOURCE_EMAIL = 'compx527brms@gmail.com'
ses = boto3.client('ses')


def message_template(anomaly, timestamp, identifier, confidence):
    date = datetime.utcfromtimestamp(int(timestamp)).isoformat()
    confidence = f'{confidence:.2f}%' if confidence != float('inf') else 'not'
    return {
        'Subject': {
            'Data': 'BAG - Anomaly Detected'
        },
        'Body': {
            'Text': {
                'Data': (f'The anomaly "{anomaly}" was detected at '
                         f'{date} (UTC) with the identifier "{identifier}".\n'
                         f'We are {confidence} sure of this anomaly.')
            }
        }
    }


def lambda_handler(event, context):
    for record in event['Records']:
        body = json.loads(record['body'])

        try:
            response = ses.send_email(
                Source=SOURCE_EMAIL,
                Destination={'ToAddresses': [body['email']]},
                Message=message_template(body['anomaly'], body['timestamp'],
                                         body['identifier'], body['confidence']))
        except:
            return {
                'statusCode': 500
            }

    return {
        'statusCode': 200
    }
