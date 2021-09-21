<?php
require 'config/aws/aws-autoloader.php';
require 'config/config.php';

use Aws\Lambda\LambdaClient;
use Aws\Lambda\Exception\LambdaException;

set_time_limit($submission_timeout);

if ($_SERVER['REQUEST_METHOD'] != 'POST' || empty($_POST['check']))
{
	http_response_code(400);
	die();
}

try
{
	$lambda = new LambdaClient($lambda_config);
	$result = $lambda->invoke([
		'FunctionName' => 'predict',
		'Payload' => $_POST['check']
	]);
}
catch (LambdaException $ex)
{
	error_log($ex);
	http_response_code(500);
	die();
}

$payload = json_decode($result['Payload']);

http_response_code($payload->statusCode);
exit($payload->body);
