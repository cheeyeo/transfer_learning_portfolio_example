## Transfer Learning


2 main kinds:

* Fine tuning

* Feature extraction

### Fine tuning

Remove head from pre-trained ConvNet ( VGG ) and replace it with new set of FC layers with random initializations

Freeze the layers below the head so the weights cannot be updated i.e. backprop doesn;t reach them


train new network with small learning rate so new set of FC Layers can learn patterns from previously learned CONV layers earlier in network

Unfreeze the rest of network and continue training to refine the process

new FC head will have fewer params than original but it depends on dataset

the new FC head is randomly initialized and connected to body of original network

If we allow the gradient from the random values in new FC layer from backpropagate to the rest of original network, we risk losing the rich discriminative features learnt by the pretrained network

hence, we freeze all earlier layers in the network and only train the new FC layers i.e. "warm up"


After the new FC Head has started to learn patterns in dataset, pause training, unfreeze the body and continue training but with SMALL LEARNING RATE i.e. don't deviate the CONV filters dramatically

training continues until sufficient accuracy is reached

Fine tuning is powerful method to obtain image classifiers from pre-trained CNNs on custom datasets


VGG 16 most commonly used for transfer learning....

Caveats:

* Require more work than feature extraction

* Choice of FC head parameters play a big role in accuracy; can't rely on regularization techniques as network has already been pre-trained and can't deviate from regularization performed by network

* For small datasets, it can be challenging to get network to start learning from cold FC start which is why we freeze the body of network first; even then the warm-up stage can be challenging and might require use of optimizers other than SGD 


* For most fine tuning problems, we are not replicating the original head of the network but to simplify it so its easier to fine tune; the less parameters in the new head network the more accurate we can be in fine tuning it for the specific task...

### 



