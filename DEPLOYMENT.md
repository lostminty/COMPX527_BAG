To deploy the solution completely, you will need to do this on a GNU/Linux
system with Docker installed. First, make sure you have installed the necessary
collections:
>$ ansible-galaxy install -r requirements.yml

Then set the following environment variables similar to something like below.
If you want to use a different region to the one specified below, then you will
have to modify `portal_src/config.php` accordingly.

>export AWS_ACCESS_KEY_ID=XXXXXXX
>
>export AWS_SECRET_ACCESS_KEY=XXXXXXX
>
>export AWS_DEFAULT_REGION=ap-southeast-2
>
>export AWS_REGION=$AWS_DEFAULT_REGION

After the AWS credentials are configured, set the below to your EC2 key pair
that you have created for deployment purposes, which should have 0600
permissions or you will get an error:

>chmod 600 /path/to/my/key_pair.pem
>
>export ANSIBLE_PRIVATE_KEY_FILE=/path/to/my/key_pair.pem
>
>export BAG_KEY_NAME=my_key_pair_name

Before you deploy, modify `lambda_src/notify.py` and set `SOURCE_EMAIL` to your
email address that sends the notification. This email address must be verified
in the AWS SES dashboard first. Note that your account must also not be in
sandbox mode for it to send emails to users without verifying them beforehand.
If you are unable to leave sandbox mode, then add your test users' email
addresses like you did with your source email address in the AWS SES dashboard
as well. The `lambda_src/predict.py` file has a variable called `MAX_NOTIFICATIONS`
which you can set to a value to keep stored notifications from growing too large.

There are more variables to edit in the two playbooks if you want to customise
the deployment further, such as dedicated tenancy for the EC2 instances (shared
by default).

Run the following to deploy most of the solution. The two playbooks can be
repeatedly run and in the rare case that something does not work on the initial
run due to e.g. a timing issue, then run them again.
>export ANSIBLE_CONFIG=./ansible.cfg
>
>ansible-playbook portal.yml prediction.yml

Finally, execute `build_and_push_predictor.sh` in `predictor_src` by first
changing directory into it. You can manually supply the `AWS_ID` environment
variable if you wish.
>cd predictor_src && ./build_and_push_predictor.sh; cd $OLDPWD

This should deploy the solution. Manually configure other users, their
permissions, and CloudWatch alarms as required by your situation.

Once you are finished, the following command offers partial cleanup. See the
comment at the top of the playbook to see what is not cleaned up by the
previous deployment playbooks and scripts.
>ansible-playbook cleanup.yml
