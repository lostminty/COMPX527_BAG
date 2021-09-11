var aws = require("aws-sdk");
var ses = new aws.SES({ region: "ap-southeast-2" });

exports.handler = async (event) => {
    
    for (const { messageId, body } of event.Records) {
        
        var params = {
            Destination: {
              ToAddresses: [body],
            },
            Message: {
              Body: {
                Text: { Data: "Hello, your prediction calculation is done" },
              },
        
              Subject: { Data: "Behaviour Anomaly Group Notification" },
            },
            Source: "compx527brms@gmail.com",
          };
        
        return ses.sendEmail(params).promise();
    }
    return `Successfully processed ${event.Records.length} messages.`;
};
