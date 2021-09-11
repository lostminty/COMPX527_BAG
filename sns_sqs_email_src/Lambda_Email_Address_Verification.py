import json
import boto3


def ses_client():
    ses = boto3.client('ses', region_name='ap-southeast-2')
    """ :type : pyboto3.ses """
    return ses


def lambda_handler(event, context):
    
    email = event['queryStringParameters']['email']
    
    response = ses_client().verify_email_address(
        EmailAddress=email
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps('Email Verification Request Has Been Sent')
    }