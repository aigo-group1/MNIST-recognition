const moongoose = require('mongoose');
const database = 'mnist';

moongoose.connect('mongodb://127.0.0.1:27017/mnist',{
        useNewUrlParser:true,
        useCreateIndex:true
    },(err)=>{
        if(err) console.log(err);
});

module.exports = moongoose;





