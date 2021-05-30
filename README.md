# Collab_MNIST

Collab MNIST is a simple test project in which I explore the question "Can independent models learn to collaborate". In this project I look into the ability for 2 models to develop the capability to exchange, receive, and decipher information to aid in their independent predictions.

The concept and implementation is simple. I'm using the MNIST database, splitting the images in half, feeding each half to the two separate models. The simplified model graph is as follows:
![Alt text](Collab_Model_Graph_Simple.jpg?raw=true "Simplified Model Graph")

## Results

### Baseline
Baseline model results using 2 Conv2D and 1 Dense layer:

Test loss: 0.0245

Test accuracy: 0.9912

### Half-Model
Second baseline, for 1 of the models using the method mentioned above, but with no collaboration or information exchange. Basically it's just the baseline model but only taking half of the image as input.

Test loss: 0.1756

Test accuracy: 0.9423

### Collaborative model
Collaborative models. Each model contains 2 Conv2D layers and 2 Dense layers. 1 of the Dense layers feeds into the other model

Test Loss: 0.1022

Left Test Loss: 0.0478

Right Test Loss: 0.0545


Left Test Accuracy: 0.9849

Right Test Accuracy: 0.9831

## Installing

Download the repository

```
git clone https://github.com/lccuellar08/Collab_MNIST.git
```

Install the requirements

```
pip install requirements.txt
```

## Running

Example:
```
python collab_model.py
```