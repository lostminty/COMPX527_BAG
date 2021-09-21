<?php
require 'config/aws/aws-autoloader.php';
require 'config/config.php';

use Aws\DynamoDb\SessionHandler;
use Aws\DynamoDb\DynamoDbClient;
use Aws\DynamoDb\Exception\DynamoDbException;
use Aws\DynamoDb\Marshaler;

function connect_to_session($db)
{
	if (session_status() == PHP_SESSION_ACTIVE)
	{
		return true;
	}

	$sessionHandler = SessionHandler::fromClient($db, [
		'table_name' => 'sessions'
	]);
	$sessionHandler->register();

	return session_start();
}

function is_logged_in()
{
	return session_status() == PHP_SESSION_ACTIVE && isset($_SESSION['EMAIL']);
}

function message($text, $args = [])
{
	$success = $args['success'] ?? false;
	$warning = $args['warning'] ?? false;
	$code = $args['code'] ?? 200;
	$ex = $args['ex'] ?? null;

	if (!$success && $code == 200)
	{
		http_response_code(400);
	}
	else if ($code != 200)
	{
		http_response_code($code);
	}

	if ($ex != null)
	{
		error_log($ex);
	}

	return ['text' => $text, 'success' => $success, 'warning' => $warning];
}

function sign_up($db, $email, $password)
{
	if (!filter_var($email, FILTER_VALIDATE_EMAIL))
	{
		return message('The email address\'s format is invalid.');
	}

	if (!preg_match($GLOBALS['password_regex'], $password))
	{
		return message($GLOBALS['password_strength_message']);
	}

	try
	{
		$result = $db->getItem([
			'TableName' => 'users',
			'Key' => ['email' => ['S' => $email]],
			'ProjectionExpression' => 'email'
		]);
	}
	catch (DynamoDBException $ex)
	{
		return message('Internal error. Try again later.', ['code' => 500, 'ex' => $ex]);
	}

	if (!empty($result['Item']))
	{
		return message('User already exists.', ['code' => 401]);
	}

	try
	{
		$new_user = $GLOBALS['new_user_prototype'];

		$new_user['email']['S'] = $email;
		$new_user['password']['S'] = password_hash($password, PASSWORD_DEFAULT);

		$db->putItem([
			'TableName' => 'users',
			'Item' => $new_user
		]);
		return message('Signed up.', ['success' => true]);
	}
	catch (Exception $ex)
	{
		return message('Internal error. Try again later.', ['code' => 500, 'ex' => $ex]);
	}
}

function sign_in($db, $email, $password)
{
	define('BAD_LOGIN_ERROR', 'Unknown email address or incorrect password.');

	try
	{
		$result = $db->getItem([
			'TableName' => 'users',
			'Key' => ['email' => ['S' => $email]],
			'ProjectionExpression' => 'password, token_limit'
		]);
	}
	catch (DynamoDBException $ex)
	{
		return message('Internal error. Try again later.', ['code' => 500, 'ex' => $ex]);
	}

	$user = $result['Item'];
	if (empty($user))
	{
		return message(BAD_LOGIN_ERROR, ['code' => 401]);
	}

	if (password_verify($password, (new Marshaler())->unmarshalValue($user['password'])))
	{
		if (!connect_to_session($db))
		{
			return message('Internal error. Try again later.', ['code' => 500, 'ex' => 'Failed to start session']);
		}

		$_SESSION['EMAIL'] = $email;
		return message('Signed in.', ['success' => true]);
	}

	// Don't explicitly tell them if the password was incorrect.
	return message(BAD_LOGIN_ERROR, ['code' => 401]);
}

function populate_token_table($db)
{
	try
	{
		$result = $db->getItem([
			'TableName' => 'users',
			'Key' => ['email' => ['S' => $_SESSION['EMAIL']]],
			'ProjectionExpression' => 'tokens'
		]);
	}
	catch (DynamoDbException $ex)
	{
		return message('Internal error. Try again later.', ['code' => 500, 'ex' => $ex]);
	}

	if (isset($result['Item']['tokens']))
	{
		$tokens = (new Marshaler())->unmarshalValue($result['Item']['tokens']);
	}
	else
	{
		$tokens = [];
	}

	if (empty($tokens))
	{
		echo '<tr scope="row"><th></th><th>&mdash;</th></tr>';
		return message('Refreshed.', ['success' => true]);
	}

	foreach ($tokens as $token)
	{
		$deletion_form = <<< END
			<form action="" method="post">
				<button class="btn btn-sm btn-danger" type="submit" name="delete_token" value="$token">
					<strong>X</strong>
				</button>
			</form>
		END;

		echo '<tr scope="row">' .
			'<th class="align-middle">' . $deletion_form . '</th>' .
			'<td>' . $token . '</td>' .
			'</tr>';
	}

	return message('Refreshed.', ['success' => true]);
}

function populate_notification_table($db)
{
	try
	{
		$result = $db->getItem([
			'TableName' => 'users',
			'Key' => ['email' => ['S' => $_SESSION['EMAIL']]],
			'ProjectionExpression' => 'notifications'
		]);
	}
	catch (DynamoDbException $ex)
	{
		return message('Internal error. Try again later.', ['code' => 500, 'ex' => $ex]);
	}

	$notifications = (new Marshaler())->unmarshalValue($result['Item']['notifications']);

	if (empty($notifications))
	{
		echo '<tr scope="row"><th>&mdash;</th><td>&mdash;</td><td>&mdash;</td></tr>';
		return message('Refreshed.', ['success' => true]);
	}

	foreach ($notifications as $notification)
	{
		echo '<tr scope="row">' .
			'<th>' . ($notification['timestamp'] ?? '&mdash;') . '</th>' .
			'<td>' . (filter_var($notification['identifier'], FILTER_SANITIZE_STRING, FILTER_FLAG_STRIP_HIGH) ?? '&mdash;') . '</td>' .
			'<td>' . (filter_var($notification['anomaly'], FILTER_SANITIZE_STRING, FILTER_FLAG_STRIP_HIGH) ?? 'Unknown') . '</td>' .
			'</tr>';
	}

	return message('Refreshed.', ['success' => true]);
}

function handle_login($db, $submission)
{
	if (is_logged_in())
	{
		return message('You are already logged in. Please log out first.', ['code' => 400]);
	}

	$submission_types = ['sign_up', 'sign_in'];
	if (!$submission || !in_array($submission, $submission_types, true))
	{
		return message('Unknown submission type.', ['code' => 400]);
	}

	$email = $_POST['email'] ?? '';
	if (!$email)
	{
		return message('No email was provided.', ['code' => 400]);
	}

	$password = $_POST['password'] ?? '';
	if (!$password)
	{
		return message('No password was provided.', ['code' => 400]);
	}

	return $submission($db, $email, $password);
}

function create_token($db)
{
	if (!is_logged_in())
	{
		return message('You are not logged in. Please log in first.', ['code' => 401]);
	}

	try
	{
		$token = bin2hex(random_bytes($GLOBALS['token_length']));
	}
	catch (Exception $ex)
	{
		return message('Could not generate new token. Try again later.', ['code' => 500, 'ex' => $ex]);
	}

	try
	{
		$result = $db->getItem([
			'TableName' => 'users',
			'Key' => ['email' => ['S' => $_SESSION['EMAIL']]],
			'ProjectionExpression' => 'token_limit'
		]);
	}
	catch (DynamoDbException $ex)
	{
		return message('Internal error. Try again later.', ['code' => 500, 'ex' => $ex]);
	}

	$token_limit = !empty($result['Item']['token_limit'])
		? (new Marshaler())->unmarshalValue($result['Item']['token_limit']) : 0;

	if (!$token_limit)
	{
		return message('You are not permitted to create any tokens.', ['code' => 401]);
	}

	try
	{
		$db->updateItem([
			'TableName' => 'users',
			'Key' => ['email' => ['S' => $_SESSION['EMAIL']]],
			'UpdateExpression' => 'ADD tokens :token',
			'ConditionExpression' => 'NOT attribute_exists(tokens) OR size(tokens) < :token_limit',
			'ExpressionAttributeValues' => [
				':token' => ['SS' => [$token]],
				':token_limit' => ['N' => (string) $token_limit]
			],
			"ReturnValues" => 'UPDATED_NEW' // TODO: Remove this.
		]);
	}
	catch (DynamoDbException $ex)
	{
		// For some reason, the "ConditionalCheckFailedException" has
		// been removed from the SDK and now only the generic
		// "DynamoDbException" remains. This is the cumbersome way to
		// check for a condition fail.
		if ($ex->getAwsErrorCode() == 'ConditionalCheckFailedException')
		{
			return message('You have reached your token limit.', ['success' => true, 'warning' => true, 'code' => 401]);
		}
		else
		{
			return message('Internal error. Try again later.', ['code' => 500, 'ex' => $ex]);
		}
	}

	return message('Successfully created token.', ['success' => true]);
}

function delete_token($db, $token)
{
	if (!is_logged_in())
	{
		return message('You are not logged in. Please log in first.', ['code' => 401]);
	}

	if (!ctype_xdigit($token))
	{
		return message('The token\'s format is incorrect.', ['code' => 400]);
	}

	try
	{
		$result = $db->updateItem([
			'TableName' => 'users',
			'Key' => ['email' => ['S' => $_SESSION['EMAIL']]],
			'UpdateExpression' => 'DELETE tokens :token',
			'ExpressionAttributeValues' => [':token' => ['SS' => [$token]]],
			'ReturnValues' => 'UPDATED_NEW'
		]);
	}
	catch (DynamoDbException $ex)
	{
		return message('Could not generate new token. Try again later.', ['code' => 500, 'ex' => $ex]);
	}

	return message('Successfully deleted token.', ['success' => true]);
}

function status_bar($status)
{
	if (empty($status) || ($status['success'] && !$status['warning']))
	{
		return;
	}

	$type = !$status['warning'] ? 'danger' : 'warning';

	return <<< END
		<div class="alert alert-$type alert-dismissible fade show rounded-0" style="margin: 0;">
			<button type="button" class="close" data-dismiss="alert">&times;</button>
			<strong>{$status['text']}</strong>
		</div>
	END;
}

function main($db)
{
	if (!empty($_POST['login']))
	{
		return handle_login($db, $_POST['login']);
	}
	else if (isset($_POST['refresh']))
	{
		return connect_to_session($db)
			? message('Refreshed.', ['success' => true])
			: message('Failed to refresh. Please sign in again.', ['code' => 401]);
	}
	else if (isset($_POST['create_token']))
	{
		return connect_to_session($db)
			? create_token($db)
			: message(
				'Internal error. Try again later.',
				['code' => 500, 'ex' => 'Session starter failed.']
			);
	}
	else if (!empty($_POST['delete_token']))
	{
		return connect_to_session($db)
			? delete_token($db, $_POST['delete_token'])
			: message(
				'Internal error. Try again later.',
				['code' => 500, 'ex' => 'Session starter failed.']
			);
	}
	else if (isset($_POST['logout']))
	{
		connect_to_session($db);
		if (is_logged_in())
		{
			session_unset();
			session_destroy();
		}

		return message('Signed out.', ['success' => true]);
	}
	else
	{
		return message('Unknown or malformed POST request.', ['code' => 400]);
	}
}

if ($_SERVER['REQUEST_METHOD'] == 'POST')
{
	try
	{
		$db = new DynamoDbClient($db_config);
	}
	catch (Exception $ex)
	{
		echo message('Internal error. Please try again later.', ['code' => 500, 'ex' => $ex])['text'];
		die();
	}

	$status = main($db);
}

?>
<!doctype html>
<html lang="en">

<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

	<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">
	<link rel="stylesheet" href="https://cdn.datatables.net/v/bs4/dt-1.11.1/datatables.min.css" />

	<link rel="stylesheet" href="portal.css">

	<title>BAG - <?php echo (!is_logged_in() ? 'Login' : $_SESSION['EMAIL']); ?></title>
</head>

<body class="text-center">
	<?php if (!is_logged_in()) : ?>
		<form class="form-signin" action="" method="post">
			<h1 class="h3 mb-3 font-weight-normal">BAG - Login</h1>
			<?php
			if (!empty($status))
			{
				echo '<div class="alert alert-';
				if (!$status['success'])
				{
					echo 'danger';
				}
				else if ($status['warning'])
				{
					echo 'warning';
				}
				else
				{
					echo 'success';
				}
				echo '" role="alert">' . $status['text'] . '</div>';
			}
			?>
			<label for="inputEmail" class="sr-only">Email Address</label>
			<input class="form-control" type="email" name="email" placeholder="Email Address" value="<?php echo $_POST["email"] ?? ''; ?>" required autofocus>
			<label for="inputPassword" class="sr-only">Password</label>
			<input class="form-control" type="password" name="password" placeholder="Password" value="<?php echo $_POST["password"] ?? ''; ?>" required>
			<button class="btn btn-lg btn-primary btn-block" type="submit" name="login" value="sign_up">Sign Up</button>
			<button class="btn btn-lg btn-primary btn-block" type="submit" name="login" value="sign_in">Sign In</button>
		</form>
	<?php else : ?>
		<nav class="navbar navbar-dark bg-dark sticky-top">
			<a class="navbar-brand" href="#">BAG - <?php echo $_SESSION["EMAIL"]; ?></a>
			<ul class="navbar-nav ml-auto">
				<li class="nav-item text-nowrap">
					<form class="form-inline" action="" method="post">
						<button class="btn btn-primary" type="submit" name="refresh">Refresh</button>
						<button class="btn btn-danger ml-3" type="submit" name="logout">Sign Out</button>
					</form>
				</li>
			</ul>
		</nav>

		<?php $bars = [status_bar($status)]; ?>

		<div class="container" style="max-width: 100%; margin: auto; padding: 0;">
			<div class="table-responsive">
				<table class="table table-bordered table-striped table-hover">
					<thead>
						<tr>
							<th scope="col">
								<form action="" method="post">
									<button class="btn btn-sm btn-success" type="submit" name="create_token">
										<strong>+</strong>
									</button>
								</form>
							</th>
							<th scope="col" style="width: 90%;"><strong>Token</strong></th>
						</tr>
					</thead>
					<tbody>
						<?php
						$status = populate_token_table($db);
						$bars[] = status_bar($status);
						?>
					</tbody>
				</table>
			</div>

			<div class="table-responsive">
				<table class="table table-bordered table-striped table-hover" id="notifications_table">
					<thead>
						<tr>
							<th scope="col">Timestamp</th>
							<th scope="col">Source Identifier</th>
							<th scope="col">Anomaly</th>
						</tr>
					</thead>
					<tbody>
						<?php
						$status = populate_notification_table($db);
						$bars[] = status_bar($status);
						?>
					</tbody>
				</table>
			</div>
		</div>

		<div id="message_footer">
			<?php
			foreach ($bars as $bar)
			{
				echo $bar;
			}
			?>
		</div>
	<?php endif; ?>
</body>

<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
<script src="https://cdn.datatables.net/v/bs4/dt-1.11.1/datatables.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js" integrity="sha384-9/reFTGAW83EW2RDu2S0VKaIzap3H66lZH81PoYlFhbGU+6BZp6G7niu735Sk7lN" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/js/bootstrap.min.js" integrity="sha384-w1Q4orYjBQndcko6MimVbzY0tgp4pWB4lZ7lr30WKz0vr/aWKhXdBNmNb5D92v7s" crossorigin="anonymous"></script>

<script>
	$(document).ready(function() {
		$('#notifications_table').DataTable({
			"columns": [{
					"type": "date",
					"render": function(timestamp) {
						let value = parseInt(timestamp, 10);
						if (value == NaN) {
							return null;
						}

						return (new Date(value * 1000)).toUTCString();
					}
				},
				{
					"searchable": true
				},
				{
					"searchable": true
				},
			]
		});
		$('.dataTables_length').addClass('bs-select');
	});
</script>

</html>