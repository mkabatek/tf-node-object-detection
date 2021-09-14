const tfn = require("@tensorflow/tfjs-node");
const tf = require('@tensorflow/tfjs');

const Canvas = require('canvas')
const fs = require('fs');

const mobilenet = require('@tensorflow-models/mobilenet');
const cocoSsd = require('@tensorflow-models/coco-ssd');

const fileName = 'antenna_image'
const extention = '.jpg'

let lite_mobilenet_v2 = 'lite_mobilenet_v2'
let mobilenet_v1 = 'mobilenet_v1'
let mobilenet_v2 = 'mobilenet_v2'

function loadLocalImage() {
  try {
    const buf = fs.readFileSync(fileName + extention)

    const input = tfn.node.decodeJpeg(buf)
    cocoSsd.load({modelUrl: 'https://storage.googleapis.com/tfjs-examples/simple-object-detection/dist/object_detection_model/model.json'}).then((model) => {
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

function runMobileNet(){

  const imageBuffer = fs.readFileSync(fileName + extention);
  const tfimage = tfn.node.decodeImage(imageBuffer);

  mobilenet.load().then((model) => {
    // detect objects in the image.
    model.classify(tfimage).then((predictions) => {
      console.log(predictions)
    })
  })

}

async function runTrainedModelExample() {
  const LOCAL_MODEL_PATH = '/Users/lol/Development/rms/tf-node-object-detection/model.json';
  const HOSTED_MODEL_PATH = 'https://sagemaker-us-west-2-603523224449.s3.us-west-2.amazonaws.com/tf2-object-detection-2021-07-07-15-30-09-356/my_tfjs_model_2/model.json';
  const buf = fs.readFileSync(fileName + extention)
  let input = tfn.node.decodeJpeg(buf).expandDims(0)


  console.log(input)
  let width = input.shape[1]
  let height = input.shape[1]
  console.log(`input width ${width}`)
  console.log(`input height ${height}`)

  let classesDir = {
    1: { name: "antenna", id: 1 },
    2: { name: "climbing_peg", id: 2 },
  };
  
  //const image = tf.image.decode_jpeg(tf.read_file(fileName + extention), channels=1)
  let model = await tf.loadGraphModel(HOSTED_MODEL_PATH);
  //model.summary();
  console.log(model.outputNodes)
  const predictions = await model.executeAsync(input);
  console.log(predictions)
  console.log(predictions.length)
  

  //Getting predictions
  const boxes = predictions[5].arraySync();
  const scores = predictions[1].dataSync();
  const classes = predictions[7].dataSync();
  //const detections = predictions[5].arraySync();
  console.log(boxes)
  console.log(scores)
  console.log(classes)
  //console.log(detections)

  const detectionObjects = [];

  scores.forEach((score, i) => {
    if (score > 0.3) {
      const bbox = [];
      const minY = boxes[0][i][0] * height;
      const minX = boxes[0][i][1] * width;
      const maxY = boxes[0][i][2] * height;
      const maxX = boxes[0][i][3] * width;
      bbox[0] = minX;
      bbox[1] = minY;
      bbox[2] = maxX - minX;
      bbox[3] = maxY - minY;

      detectionObjects.push({
        class: classes[i],
        label: classesDir[classes[i]].name,
        score: score.toFixed(4),
        bbox: bbox
      });
    }
  });

  console.log(detectionObjects)

}

//runMobileNet()
//loadLocalImage()
runTrainedModelExample()


/**
 * This file runs inference on a pretrained simple object detection model.
 *
 * The model is defined and trained with `train.js`.
 * The data used for model training and model inference are synthesized
 * programmatically. See `synthetic_images.js` for details.
 */

// import * as tf from '@tensorflow/tfjs';
// import {ObjectDetectionImageSynthesizer} from './synthetic_images';

// const canvas = document.getElementById('data-canvas');
// const status = document.getElementById('status');
// const testModel = document.getElementById('test');
// const loadHostedModel = document.getElementById('load-hosted-model');
// const inferenceTimeMs = document.getElementById('inference-time-ms');
// const trueObjectClass = document.getElementById('true-object-class');
// const predictedObjectClass = document.getElementById('predicted-object-class');

// const TRUE_BOUNDING_BOX_LINE_WIDTH = 2;
// const TRUE_BOUNDING_BOX_STYLE = 'rgb(255,0,0)';
// const PREDICT_BOUNDING_BOX_LINE_WIDTH = 2;
// const PREDICT_BOUNDING_BOX_STYLE = 'rgb(0,0,255)';

// function drawBoundingBoxes(canvas, trueBoundingBox, predictBoundingBox) {
//   tf.util.assert(
//       trueBoundingBox != null && trueBoundingBox.length === 4,
//       `Expected boundingBoxArray to have length 4, ` +
//           `but got ${trueBoundingBox} instead`);
//   tf.util.assert(
//       predictBoundingBox != null && predictBoundingBox.length === 4,
//       `Expected boundingBoxArray to have length 4, ` +
//           `but got ${trueBoundingBox} instead`);

//   let left = trueBoundingBox[0];
//   let right = trueBoundingBox[1];
//   let top = trueBoundingBox[2];
//   let bottom = trueBoundingBox[3];

//   const ctx = canvas.getContext('2d');
//   ctx.beginPath();
//   ctx.strokeStyle = TRUE_BOUNDING_BOX_STYLE;
//   ctx.lineWidth = TRUE_BOUNDING_BOX_LINE_WIDTH;
//   ctx.moveTo(left, top);
//   ctx.lineTo(right, top);
//   ctx.lineTo(right, bottom);
//   ctx.lineTo(left, bottom);
//   ctx.lineTo(left, top);
//   ctx.stroke();

//   ctx.font = '15px Arial';
//   ctx.fillStyle = TRUE_BOUNDING_BOX_STYLE;
//   ctx.fillText('true', left, top);

//   left = predictBoundingBox[0];
//   right = predictBoundingBox[1];
//   top = predictBoundingBox[2];
//   bottom = predictBoundingBox[3];

//   ctx.beginPath();
//   ctx.strokeStyle = PREDICT_BOUNDING_BOX_STYLE;
//   ctx.lineWidth = PREDICT_BOUNDING_BOX_LINE_WIDTH;
//   ctx.moveTo(left, top);
//   ctx.lineTo(right, top);
//   ctx.lineTo(right, bottom);
//   ctx.lineTo(left, bottom);
//   ctx.lineTo(left, top);
//   ctx.stroke();

//   ctx.font = '15px Arial';
//   ctx.fillStyle = PREDICT_BOUNDING_BOX_STYLE;
//   ctx.fillText('predicted', left, bottom);
// }

// /**
//  * Synthesize an input image, run inference on it and visualize the results.
//  *
//  * @param {tf.Model} model Model to be used for inference.
//  */
// async function runAndVisualizeInference(model) {
//   // Synthesize an input image and show it in the canvas.
//   const synth = new ObjectDetectionImageSynthesizer(canvas, tf);

//   const numExamples = 1;
//   const numCircles = 10;
//   const numLineSegments = 10;
//   const {images, targets} = await synth.generateExampleBatch(
//       numExamples, numCircles, numLineSegments);

//   const t0 = tf.util.now();
//   // Runs inference with the model.
//   const modelOut = await model.predict(images).data();
//   inferenceTimeMs.textContent = `${(tf.util.now() - t0).toFixed(1)}`;

//   // Visualize the true and predicted bounding boxes.
//   const targetsArray = Array.from(await targets.data());
//   const boundingBoxArray = targetsArray.slice(1);
//   drawBoundingBoxes(canvas, boundingBoxArray, modelOut.slice(1));

//   // Display the true and predict object classes.
//   const trueClassName = targetsArray[0] > 0 ? 'rectangle' : 'triangle';
//   trueObjectClass.textContent = trueClassName;

//   // The model predicts a number to indicate the predicted class
//   // of the object. It is trained to predict 0 for triangle and
//   // 224 (canvas.width) for rectangel. This is how the model combines
//   // the class loss with the bounding-box loss to form a single loss
//   // value. Therefore, at inference time, we threshold the number
//   // by half of 224 (canvas.width).
//   const shapeClassificationThreshold = canvas.width / 2;
//   const predictClassName =
//       (modelOut[0] > shapeClassificationThreshold) ? 'rectangle' : 'triangle';
//   predictedObjectClass.textContent = predictClassName;

//   if (predictClassName === trueClassName) {
//     predictedObjectClass.classList.remove('shape-class-wrong');
//     predictedObjectClass.classList.add('shape-class-correct');
//   } else {
//     predictedObjectClass.classList.remove('shape-class-correct');
//     predictedObjectClass.classList.add('shape-class-wrong');
//   }

//   // Tensor memory cleanup.
//   tf.dispose([images, targets]);
// }

// async function init() {
//   const LOCAL_MODEL_PATH = 'object_detection_model/model.json';
//   const HOSTED_MODEL_PATH =
//       'https://storage.googleapis.com/tfjs-examples/simple-object-detection/dist/object_detection_model/model.json';

//   // Attempt to load locally-saved model. If it fails, activate the
//   // "Load hosted model" button.
//   let model;
//   try {
//     model = await tf.loadLayersModel(LOCAL_MODEL_PATH);
//     model.summary();
//     testModel.disabled = false;
//     status.textContent = 'Loaded locally-saved model! Now click "Test Model".';
//     runAndVisualizeInference(model);
//   } catch (err) {
//     status.textContent = 'Failed to load locally-saved model. ' +
//         'Please click "Load Hosted Model"';
//     loadHostedModel.disabled = false;
//   }

//   loadHostedModel.addEventListener('click', async () => {
//     try {
//       status.textContent = `Loading hosted model from ${HOSTED_MODEL_PATH} ...`;
//       model = await tf.loadLayersModel(HOSTED_MODEL_PATH);
//       model.summary();
//       loadHostedModel.disabled = true;
//       testModel.disabled = false;
//       status.textContent =
//           `Loaded hosted model successfully. Now click "Test Model".`;
//       runAndVisualizeInference(model);
//     } catch (err) {
//       status.textContent =
//           `Failed to load hosted model from ${HOSTED_MODEL_PATH}`;
//     }
//   });

//   testModel.addEventListener('click', () => runAndVisualizeInference(model));
// }

// init();



