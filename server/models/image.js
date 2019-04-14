const moongoose = require('../mongoose');
const Image = moongoose.model('Image',{
    imagePath:{
        type:String,
        require:true
    } 
});

module.exports = Image;