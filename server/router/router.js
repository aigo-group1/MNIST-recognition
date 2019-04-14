const express = require('express');
const ImageController = require('../controllers/ImageController')
const router = express.Router();
const multer = require('multer');

const storage = multer.diskStorage({
    destination:(req,file,cb)=>{
        cb(undefined,'./upload')
    },
    filename:(req,file,cb)=>{
        cb(undefined,file.originalname)
    }
})
const upload = multer({
    storage:storage,
    limits:{
        fileSize:2000000,
    },
    fileFilter(req,file,cb){
        if(!file.originalname.match(/\.(jpg|jpeg|png)$/)){
            cb(new Error('Please Upload Image File'),false);
        }
        else if(file.size >2000000){
            cb(new Error('The size of file is too large'),false)
        }
        else cb(undefined,true);
    }
});

router.get('/',(req,res)=>{
    res.render('index');
})
router.post('/',upload.single('image'),ImageController.uploadImage,(err,req,res,next)=>{
    if(err){
        res.status(400).send({error:err.message});
    }
});
module.exports = router;