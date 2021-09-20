from datetime import datetime
import json
import boto3

ses = boto3.client('ses')

def message_template(anomaly, timestamp, identifier):
    date = datetime.fromtimestamp(int(timestamp)).isoformat()
    return {
        'Subject': {
            'Data': 'BAG - Anomaly Detected'
        },
        'Body': {
            'Text': {
                'Data': f'The anomaly "{anomaly}" was detected at ' +
                    f'{date} (UTC) with the identifier "{identifier}".'
            }
        }
    }

def lambda_handler(event, context):
    for record in event['Records']:
        body = json.loads(record['body'])
        
        email = body['email']
        anomaly = body['anomaly']
        timestamp = body['timestamp']
        identifier = body['identifier']
        
        try:
            response = ses.send_email(
                Source='compx527brms@gmail.com',
                Destination={'ToAddresses': [body['email']]},
                Message=message_template(anomaly, timestamp, identifier))
        except:
            return {
                'statusCode': 500
            }

    return {
        'statusCode': 200
    }
