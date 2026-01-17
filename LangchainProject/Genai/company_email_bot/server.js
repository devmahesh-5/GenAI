import express from 'express'
import { createServer } from 'http'
import nodemailer_mail from './utils/nodemailer.js'
import path from 'path'
import cors from 'cors'
import dotenv from 'dotenv'

dotenv.config({ path: './.env' })
const app = express()
const port = 3000;
app.use(express.json())
app.use(express.urlencoded({ extended: true }))
app.use(cors())

const company_data = [
    {
        "name" : "Mahesh Bhandari",
        "email" : "n31mahesh@gmail.com",
        "role" : "CEO"
    },
    {
        "name" : "Hari Bhandari",
        "email" : "maheshbhandari2061@gmail.com",
        "role" : "Manager"
    },
    {
        "name" : "Ramesh Bhandari",
        "email" : "info.sikshyakendra@gmail.com",
        "role" : "Sales Executive"
    }
]

app.use(express.static(path.join(path.resolve(), 'public')))
app.get('/users',(req,res)=>{
    
    const {role, name} = req.query;
    if (!role.trim() && !name.trim()) {
        return res.status(400).json({error : "Please provide role or name"});
    }
    
    const filteredData = company_data.filter((user) => user.role == role || user.name == name);

    if (filteredData.length === 0) {
        return res.status(404).json({error : "No data found"});
    }
    res.status(200).json(filteredData);
})

app.post('/users/send-email', async(req, res) => {
    ;(
        async () =>{
            const {to, subject, body,from,pass} = req.body;
    try {
        if (!to || !subject || !body || !from || !pass) {
            return res.status(400).json({error : "Please provide to, subject and message"});
        }
        const result = await nodemailer_mail(from,pass,subject,body,to);
        // console.log(result);
        // if (!result) {
        //     return res.status(500).json({error : "Something went wrong"});
        // }

        res.status(200).json({message : "Email sent successfully"});
    } catch (error) {
        res.status(500).json({error : "Something went wrong"});
    }
        }
    )();
})

const server = createServer(app)
server.listen(port, () => {
    console.log(`Server is running on port ${port}`);
})

