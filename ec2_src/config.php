<?php

// Debug information. Left off.
// error_reporting(E_ALL);
// ini_set('display_errors', 'On');

// Set the following regions as you need.

$db_config = [
	'version' => '2012-08-10',
	'region' => 'ap-southeast-2',
	// 'endpoint' => 'http://localhost:8000', // For local development/testing.
];

$lambda_config = [
	'version' => 'latest',
	'region' => 'ap-southeast-2',
];

$password_regex = '#.*^(?=.{8,64})(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9]).*$#';
$password_strength_message = <<< 'END'
	The password must be between 8 to 64 characters long
	with a mix of digits, lowercase letters, and uppercase letters.
END;

$new_user_prototype = [
	'email' => [],
	'password' => [],
	'credit' => ['N' => '0'],
	'token_limit' => ['N' => '5'],
	'notifications' => ['L' => []]
];

$token_length = 24;
