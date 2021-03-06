Finding Similar Images
======================

This is a learning repository for a neural network that finds similar images.

Training the Model
------------------

We assume jpeg images of size 512x512 are store in a folder.
The trained model and the embedding array are saved to disk and made
available for testing.

```bash
python main.py --mode train \
               --train-data "data/train"
```

Testing the Model
-----------------

Testing the model involves reading in the trained model and embedding matrix,
we then use it to find similar images in the training data for the images
available in the testing data directory.

```bash
python main.py --mode test \
               --train-data "data/train" \
               --test-data "data/test" \
               --num-similar-imgs 9
```

Sample output, the first image in first row represents the image for which
similar images are found.

<img src="./images/three.png" width=500>
