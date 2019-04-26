const Image = require('../models/image');
const path = require('path');
const uploadPath = path.join(__dirname,'/../');
const {spawn} = require('child_process');

const request  = require('request');

const pythonServerUrl = 'http://127.0.0.1:5000/predict/'; 

/*
predict = (modelPath,imagePath)=>{
    return new Promise((get,drop)=>{
        const pyProg = spawn('python', [modelPath,imagePath]);
        pyProg.stdout.on('data',(data)=>{
            get(data.toString())
        })
        pyProg.stderr.on('err',(err)=>{
            drop(err.toString())
        })
        pyProg.on('close', (code) => {
            console.log(`child process exited with code ${code}`);
        });
    })
}
uploadImage = async (req,res)=>{

    const filePath = req.file.path;

    const pathToUpload = path.join(uploadPath,filePath);
    
    const image = new Image({imagePath:filePath});

    const result = await image.save();

    predict('../Predict_image.py',pathToUpload).then((data)=>{

        console.log(data)
       const arr = data.split('\n');
       const result = arr.map((value)=>{
           return value.trim();
       })
       const obj = result.map((value)=>{
           const temp = value.split('');
           return temp;
       })
       obj2 = [];
       obj.pop();
       if(obj.length % 2 == 0 && obj.length>0){
            for(var i =0;i<=obj.length-2;i+=2){
                obj2.push({cmnd:obj[i],birthday:obj[i+1]})
            }
       }
       else if(obj.length %2 !=0 &&obj.length >0) {
           obj.push([]);
           for(var i =0;i<=obj.length-2;i+=2){
                obj2.push({cmnd:obj[i],birthday:obj[i+1]})
            }
       }
       res.render('predict',{obj2});
    }).catch((err)=>{
        console.log(err)
    }) 
};
*/

uploadImage = async (req,res)=>{

    const filePath = req.file.path;

    const imagepath = filePath.toString().split('/')[1]

    const pathToUpload = path.join(uploadPath,filePath);
    
    const image = new Image({imagePath:filePath});

    const result = await image.save();


    request({uri:pythonServerUrl + imagepath},(err,req2)=>{
        if(err) console.log(err);
        else{
            obj = JSON.parse(req2.body)
            console.log(req2.body)
            res.render('predict',{obj})
        }
    })

};
module.exports = {
    uploadImage:uploadImage
}