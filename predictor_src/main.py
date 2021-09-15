import sys
import torch
import torchvision


def classify():
    return "<TODO: classification>"


def lambda_handler(event, context):
    message = classify()

    return {
        "message": message
    }
