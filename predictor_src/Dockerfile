FROM public.ecr.aws/lambda/python:3.9

# Don't install these from the regular source to save on a lot of time and
# space, as AWS Lambda seemingly can't use GPU acceleration.
RUN pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html --target "${LAMBDA_TASK_ROOT}"

COPY main.py ${LAMBDA_TASK_ROOT}
CMD [ "main.lambda_handler" ]

# Copy any other files as necessary.
# COPY model.pt .
