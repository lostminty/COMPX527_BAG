import boto3
import calendar
import time


def sns_client():
    sns = boto3.client('sns', region_name='us-east-1')
    """ :type : pyboto3.sns """
    return sns

def sqs_client():
    sqs = boto3.client('sqs', region_name='us-east-1')
    """ :type : pyboto3.sqs """
    return sqs


def create_email_subscription(topic_arn, email_address):
    return sns_client().subscribe(
        TopicArn=topic_arn,
        Protocol='email',
        Endpoint=email_address
    )

def create_sqs_queue_subscription(topic_arn, queue_url):
    return sns_client().subscribe(
        TopicArn=topic_arn,
        Protocol='sqs',
        Endpoint=queue_url
    )

def send_message_to_queue(queue_url, email_address):
    return sqs_client().send_message(
        QueueUrl=queue_url,
        MessageAttributes={
            'Email': {
                'DataType': 'String',
                'StringValue': email_address
            }
        },
        MessageBody='SQS Message for Email'
    )

def publish_message(topic_arn):
    return sns_client().publish(
        TopicArn=topic_arn,
        Message="Hello, your prediction calculation is done. The result is "
    )


if __name__ == '__main__':

    QUEUE_NAME = "QUEUE_NAME.fifo"
    AWS_ACCOUNT_ID = "XXXXXXXXXXXX"

    USER_EMAIL_ADDRESS = "XXX@XXX"

    ts = calendar.timegm(time.gmtime())
    responseSNS = sns_client().create_topic(Name=ts)
    TOPIC_ARN = responseSNS["TopicArn"]

    responseSQS = sqs_client().get_queue_url(QueueName=QUEUE_NAME, QueueOwnerAWSAccountId=AWS_ACCOUNT_ID)
    QUEUE_URL = responseSQS["QueueUrl"]
    

    create_email_subscription(TOPIC_ARN, USER_EMAIL_ADDRESS)

    create_sqs_queue_subscription(TOPIC_ARN, QUEUE_URL)

    send_message_to_queue(QUEUE_URL, USER_EMAIL_ADDRESS)

    publish_message(TOPIC_ARN)

