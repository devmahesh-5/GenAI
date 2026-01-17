import nodemailer from "nodemailer";
// import dotenv from 'dotenv'
// dotenv.config()
// Create a transporter using Ethereal test credentials.
// For production, replace with your actual SMTP server details.
const nodemailer_mail = (user,pass,subject,body,to) => {
    console.log(user,pass,subject,body,to);
    const transporter = nodemailer.createTransport({
    host: 'smtp.gmail.com',
    port: 587, // SSL
    secure: false,
    requireTLS: true,
    auth: {
        user:user,
        pass:pass,
    },
});

// Send an email using async/await
let info;
(async () => {
  let info = await transporter.sendMail({
    from: user,
    to: `${to}`,
    subject: subject,
    text: body, // Plain-text version of the message
    html: "<b>" + body + "</b>", // HTML version of the message
  });

  console.log("Message sent:", info.messageId);
  
})();
return info
}

export default nodemailer_mail