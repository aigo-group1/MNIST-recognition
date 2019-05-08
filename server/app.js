const express = require('express');
const app = express();
const path = require('path');
const publicPath = path.join(__dirname,'./public');
const bodyParser = require('body-parser');
const hbs = require('hbs');

const viewPath = path.join(__dirname,'./views');
const uploadPath = path.join(__dirname,'');
const router = require('./router/router');

//console.log(publicPath);
//console.log(viewPath);
app.use(express.static(uploadPath));
app.use(express.json());
app.use(express.static(publicPath));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended:false}));

app.set('view engine','hbs');
app.set('views',viewPath);



const cached = require('memcached');

const memcached = new cached("127.0.0.1:3000");

memcachedMidleware = (duration)=>{
    return (req,res,next)=>{
        const key = "__express__" + req.originalUrl || req.url;
        memcached.get(key,(err,data)=>{
            if(data){
                res.send(data);
                return;
            }
            else{
                res.sendRespone = res.send;
                res.send = (body)=>{
                    memcached.set(key,body,(duration*60),(err)=>{

                    })
                    res.sendRespone(body);
                }
                next();
            }
        })
    }
}

app.use('/',router);


app.listen(3000,()=>{
    console.log('server is in port 3000');
})