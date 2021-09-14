First, make sure you have installed the necessary collections:
>ansible-galaxy install -r requirements.yml

You may also need to export this as well if a world-writeable
directory warning appears.

>export ANSIBLE_CONFIG=./ansible.cfg

Then set the following environment variables like below:

>export AWS_ACCESS_KEY_ID=XXXXXXX
>
>export AWS_SECRET_ACCESS_KEY=XXXXXXX
>
>export AWS_DEFAULT_REGION=ap-southeast-2
>
>export AWS_REGION=$AWS_DEFAULT_REGION

After the AWS credentials are configured, set the below to
your key pair (which should have 0600 permissions or you will get an error):

>chmod 600 /path/to/my/key_pair.pem
>
>export ANSIBLE_PRIVATE_KEY_FILE=/path/to/my/key_pair.pem
>
>export BAG_KEY_NAME=my_key_pair_name

There are more variables to edit in "main.yml" if you want to customise the
deployment further.

Finally, run the following to execute the playbook:

>ansible-playbook main.yml
