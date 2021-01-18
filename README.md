Finding Similar Images
======================

This is a learning repository for a neural network that finds similar images

Training the Model
------------------

We assume MNIST jpeg images are store in a folder, the trained model
and the embedding array is saved to disk

```bash
python main.py --mode train \
							--train-data "data/train"
```

Testing the Model
-----------------

Testing the model involves reading in the trained model and embedding matrix,
we then use it to find similar images in the training data for the images
available in the testing data directory

```bash
python main.py --mode test \
							--train-data "data/train"
							--test-data "data/test"
```

Sample output, the first image in first row represents the image for which
similar images are found

[sample_plot](./imgs/three.jpg)
