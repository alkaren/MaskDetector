# MaskDetector
The Minister of Health of the Republic of Indonesia establishes a health protocol for the public in public facilities with the aim of increasing efforts to prevent and control Covid-19 in order to prevent the occurrence of new epicenters or clusters. According to the Centers for Disease Control and Prevention (CDC) the impact of not wearing a mask in public places is being easy to catch the virus, spread the virus to other people, and carry the virus into the body. People also need to return to their activities by adapting to new habits such as always wearing masks in public places. By doing the classification, you can monitor the use of masks in the context of monitoring Covid-19. This classification process implements a convolutional neural network algorithm using mobilenetv2 in its modeling architecture. In this classification, there are several stages, namely doing the CRISP-DM (Cross Industry Standard for Data Mining) method first to process the data, the data used comes from the Kaggle and Github sites. Then make three scenarios for testing the dataset in order to get the highest results to be used as a model, then to calculate the results using the confusion matrix method for accuracy calculations, from the results of the dataset test using 4000 photos, the highest accuracy value is 98.75%.

![Alt Text](https://github.com/alkaren/MaskDetector/blob/master/Poster.png)

## Technologies
- Keras/Tensorflow
- OpenCV
- Flask
- MobilenetV2

## Usage
You have to install the required packages, you can do it:
- via pip
```pip install -r requirements.txt```
- or via conda
```conda env create -f environment.yml```

Once you installed all the required packages you can type in the command line from the root folder:

```
python wsgi.py
```
and click on the link that the you will see on the prompt.

## Data
The dataset used for training the model is available
<ul>
  <li><a href="https://www.kaggle.com/omkargurav/face-mask-dataset">Face Mask</a></li>
  <li><a href="https://github.com/cabani/MaskedFace-Net">MaskedFace-Net</a></li>
  <li><a href="https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset">Real-World Masked Face Dataset，RMFD</a></li>
</ul>