const Image = require('../models/image');
const path = require('path');
const uploadPath = path.join(__dirname,'/../');
const {spawn} = require('child_process');

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

    predict('../mnist_model/readImage.py',pathToUpload).then((data)=>{
        console.log(data)
    }).catch((err)=>{
        console.log(err)
    }) 
};



module.exports = {
    uploadImage:uploadImage
}