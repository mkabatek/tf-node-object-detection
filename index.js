const tfn = require("@tensorflow/tfjs-node");
const cocoSsd = require('@tensorflow-models/coco-ssd');
const fs = require('fs');
const Canvas = require('canvas')

const fileName = 'DSC_0440'
const extention = '.jpg'

function loadLocalImage() {
  try {
    const buf = fs.readFileSync(fileName + extention)

    const input = tfn.node.decodeJpeg(buf)
    cocoSsd.load().then((model) => {
      // detect objects in the image.
      model.detect(input).then((predictions) => {
        console.log('Predictions: ', predictions);

        var img = new Canvas.Image; // Create a new Image
        img.src = buf;
        var canvas = new Canvas.createCanvas(img.width, img.height);
        var ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, img.width, img.height);
        
        for (prediction of predictions) {
          var color = '#'+(Math.random() * 0xFFFFFF << 0).toString(16).padStart(6, '0');
          ctx.beginPath();
          ctx.font = '256px blippo';
          ctx.fillStyle = color;
          ctx.fillText(prediction.class + ": " + prediction.score, prediction.bbox[0], prediction.bbox[1] - 30);
          ctx.rect(prediction.bbox[0], prediction.bbox[1], prediction.bbox[2], prediction.bbox[3]);
          ctx.strokeStyle = color;
          ctx.lineWidth = 15;
          ctx.stroke();
        }

        var img = canvas.toDataURL("image/jpeg", 0.8);
        // console.log(img)
        var data = img.replace(/^data:image\/\w+;base64,/, "");
        var buf1 = new Buffer(data, 'base64');
        fs.writeFile(fileName + '_cocoSsd' + extention, buf1, function(err, result) {
          if(err) console.log('error', err);
        });

      });
    });
    
  } catch (err) {
    console.log(err);
  }
}

loadLocalImage()


