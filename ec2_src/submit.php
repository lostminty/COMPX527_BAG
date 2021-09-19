<?php
require 'config/aws/aws-autoloader.php';
require 'config/config.php';

use Aws\Lambda\LambdaClient;
use Aws\Lambda\Exception\LambdaException;

if (
	$_SERVER['REQUEST_METHOD'] != 'POST'
	|| empty($_POST['email']) || empty($_POST['token'])
	|| empty($_POST['identifier']) || empty($_POST['image'])
)
{
	http_response_code(400);
	die();
}

try
{
	$lambda = new LambdaClient($lambda_config);
}
catch (LambdaException $ex)
{
	error_log($ex);
	http_response_code(500);
	die();
}

$payload = json_encode([
	'email' => $_POST['email'],
	'token' => $_POST['token'],
	'identifier' => $_POST['identifier'],
	'image' => $_POST['image']
]);

if (!$payload)
{
	http_response_code(400);
	die();
}

try
{
	$result = $lambda->invoke([
		'FunctionName' => 'predict',
		'Payload' => $payload
	]);
}
catch (LambdaException $ex)
{
	error_log($ex);
	http_response_code(500);
	die();
}

http_response_code($result['StatusCode']);
exit($result['Payload']);
