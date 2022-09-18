# Bag of Visual Words

This repository presents the image classification using Bag of Visual Words (BoVW) and Extreme Learning Machine (ELM) classifier. The data used are the faces of three celebrities, namely, Arnold Schwarzenegger, Silvester Stallone, and Taylor Swift. The [training](data/training/) data has 25 images of each label. To test the built model, 3 [test](data/test/) images of each label are used.

Two modules are created [bovw](src/bovw/) and [elm](src/elm/). The [bovw](src/bovw/) is used to build the BoVW to change the image representation into a new and better one. Meanwhile, the [elm](src/elm/) is used to build the classifier. The ELM approach was used previously on GitHub and applied to time series forecasting and Iris plant classification.

The [notebook](notebooks/bovw-classification.ipynb) is used to investigate image classification using BoVW and ELM classifier. The BoVW and ELM models are saved in [models](models/) directory.

---
Useful links:
- BoVW classification (blog post): [Medium](https://rlrocha.medium.com/bag-of-visual-words-applied-to-image-classification-64a7de0b6369).
- BoVW classification App: [Hugging Face](https://huggingface.co/spaces/rlrocha/bovw-classification).
- ELM GitHub repository: [GitHub](https://github.com/rlrocha/elm).
- ELM to time series forecasting (blog post): [Medium](https://rlrocha.medium.com/time-series-forecasting-through-extreme-learning-machine-b6fa5917ebbb).
- ELM to Iris plant classification (blog post): [Medium](https://rlrocha.medium.com/extreme-learning-machine-to-multiclass-classification-cf9d4fe34b40).