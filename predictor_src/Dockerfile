FROM public.ecr.aws/lambda/python:3.9

# Don't install these from the regular source to save on a lot of time and
# space, as AWS Lambda seemingly can't use GPU acceleration.
# RUN pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html --target "${LAMBDA_TASK_ROOT}"
# PyTorch-Lightning drags in the full packages, unfortunately.
RUN pip3 install pytorch-lightning torch torchvision scikit-metrics --target "${LAMBDA_TASK_ROOT}"

# Copy any other files as necessary.
COPY model.ckpt .

COPY litautoencoder.py ${LAMBDA_TASK_ROOT}
COPY predictor.py ${LAMBDA_TASK_ROOT}
CMD [ "predictor.lambda_handler" ]
