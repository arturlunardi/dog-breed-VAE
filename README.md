# About

This application provides an implementation of variational auto-encoder [(VAE)](https://arxiv.org/abs/1906.02691) for [Dog Breed Dataset](https://www.kaggle.com/c/dog-breed-identification/data).

VAE it is a unsupervised model, so the model it was builded to be trained in both train and test data from the dataset.

# Reproduce

1. Clone the repository

2. Download the data

Access the [Dog Breed Dataset](https://www.kaggle.com/c/dog-breed-identification/data), download the data and extract it on the data directory.

3. Run and save the model

Access the project directory and install the requirements.

```
pip install -r requirements.txt
```

Go to the application directory, run the model file to fit and save it.

```
python model.py
```

4. Evaluate images

To generate or evaluate images, access the predict_images notebook in the application directory.

![vae_output](https://i.ibb.co/F55W9Tv/output.png)