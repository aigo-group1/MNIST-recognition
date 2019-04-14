const Image = require('../models/image');

uploadImage = async (req,res)=>{
    const name = req.file;
    const filePath = req.file.path;
    const image = new Image({imagePath:filePath});
    const result = await image.save();
    console.log(result);
};

module.exports = {
    uploadImage:uploadImage
}