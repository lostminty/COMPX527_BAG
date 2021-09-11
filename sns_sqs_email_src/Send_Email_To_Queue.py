import json
import boto3

def lambda_handler(event, context):

    return boto3.client('sqs', region_name='ap-southeast-2').send_message(
        QueueUrl='https://sqs.ap-southeast-2.amazonaws.com/343109335303/Queue_User_Notification',
        MessageBody='UserEmail@Example.com'
    )