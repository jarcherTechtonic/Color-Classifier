let data;
let model;
let x_values, y_values;
let lossP;
let epochP;
let labelP;
let rSlider, gSlider, bSlider;
let labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish'
];

function preload() {
  data = loadJSON('colorData.json');
}

function setup() {
  // console.log(data.entries.length); // useful for knowing amount of data entries in your sample
  labelP  = createP('Label:');
  line("\n");
  epochP  = createP('Epoch:')
  line("\n");
  lossP   = createP('Loss:');
  line("\n");
  
  rSlider = createSlider(0, 255, 255);
  gSlider = createSlider(0, 255, 255);
  bSlider = createSlider(0, 255, 0);
  createP('This is a machine learning project that classifies colors.');
  createP('When the program starts, the AI starts its training with a large batch of data--over 70,000 JSON color objects!');
  createP("The guess will be incorrect at first, but over time it will \"learn\" what the right color is.");
  createP("Every epoch is equivalent to going over every single entry once. It starts at zero because computers.");
  createP("Move the sliders around to change the color to watch the AI change its guess.");
  createP("It will only get more accurate over time.");

  let colors = [];
  let labels = [];
  for (let record of data.entries) {
    let color = [record.r / 255, record.g / 255, record.b / 255,];
    colors.push(color);
    labels.push(labelList.indexOf(record.label));
  }

  x_values = tf.tensor2d(colors);

  let labelsTensor = tf.tensor1d(labels, 'int32');

  y_values = tf.oneHot(labelsTensor, 9);
  labelsTensor.dispose();

  // Building the model //
  model = tf.sequential();

  let hidden = tf.layers.dense({
    units: 16,
    activation: 'sigmoid',
    inputDim: 3
  })
  let output = tf.layers.dense({
    units: 9,
    activation: 'softmax'
  })
  model.add(hidden);
  model.add(output);

  // Create an optimizer function, sgd = stochastic gradient descent 
  const learningRate = 0.2; // the learning rate determines how "large" the corrections are
  const optimizer = tf.train.sgd(learningRate);
  // compile the model
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy'
  })

  trainModel().then(results => {
    console.log(results.history.loss);
  });

}

async function trainModel() {
  const options = {
    epochs: 10,
    validationSplit: 0.1, // this removes 10 percent of the data at the start of each epoch
    shuffle: true, // this shuffles the data around so it isn't processed in the same order again
    callbacks: {
      onTrainBegin: () =>   console.log('training started'),
      onTrainEnd:   () => console.log('training completed'),
      onBatchEnd: tf.nextFrame,
      onEpochEnd : (num, logs) => {
        epochP.html('Epoch: ' + num);
        lossP.html('Loss: ' + logs.loss);
      }
    }
  }
  return await model.fit(x_values, y_values, options);
}

function draw() {
  let r = rSlider.value();
  let g = gSlider.value();
  let b = bSlider.value();
  background(r, g, b);

  tf.tidy(() => {
    const inputXs = tf.tensor2d([
      [r / 255, g / 255, b / 255]
    ]);
    let results = model.predict(inputXs);
    let index   = results.argMax(1).dataSync()[0];
    let label   = labelList[index];
    labelP.html("Color: " + label);
  })
}

