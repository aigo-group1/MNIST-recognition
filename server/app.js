const express = require('express');
const app = express();
const path = require('path');
const publicPath = path.join(__dirname,'./public');
const bodyParser = require('body-parser');
const hbs = require('hbs');

const viewPath = path.join(__dirname,'./views');

const router = require('./router/router');

console.log(publicPath);
console.log(viewPath);

app.use(express.json());
app.use(express.static(publicPath));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended:false}));

app.set('view engine','hbs');
app.set('views',viewPath);


app.use('/',router);

app.listen(3000,()=>{
    console.log('server is in port 3000');
})
