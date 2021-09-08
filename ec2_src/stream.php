<?php
require 'aws/aws-autoloader.php';
require 'config.php';

use Aws\Lambda\LambdaClient;
use Aws\Lambda\Exception\LambdaException;

if (
	$_SERVER['REQUEST_METHOD'] != 'POST'
	|| empty($_POST['e']) || empty($_POST['t'])
	|| empty($_POST['i']) || empty($_POST['d'])
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
	'email' => $_POST['e'],
	'token' => $_POST['t'],
	'identifier' => $_POST['i'],
	'image' => $_POST['d']
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
