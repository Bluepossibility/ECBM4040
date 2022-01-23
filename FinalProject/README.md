# e4040-2021Fall-project
This is the README file of Group CSGO Project: Spectral Representations for Convolutional Neural Networks. The project repo is a reproduction of Rippel, Snoek, and Adams 2015 (https://arxiv.org/pdf/1506.03767.pdf). 

All files containing code are located in the src folder. Our projects consist of 6 Jupyter Notebooks and 8 modules, i.e. python files. 6 Jupiter Notebooks are respectively named Approximation Loss.ipynb, Spectral Parameterization.ipynb, Hyperparameter Search.ipynb, Information Preserved.ipynb, CIFAR100 Train Test.ipynb and CIFAR10 Train Test.ipynb. The overall structure is displayed in the figure as follows.

<img width="930" alt="bdff417939090e2728d44353ed5036a" src="https://user-images.githubusercontent.com/90977332/147037998-d2ac6586-83b5-45f9-a49c-68230e0d03e8.png">

Approximation Loss.ipynb - This Jupyter Notebook demonstrates spectral pooling and frequency dropout in action on a minibatch. It also replicates the results for the approximation loss from the original paper.

Spectral Parameterization.ipynb - This Jupyter Notebook replicates the comparison of convergence time (measured in epochs) for traditionally vs spectrally parameterized CNNs.

Information Preserved.ipynb - This Jupyter Notebook compares the spectral pooling and max pooling in the aspect of image compression and information preservation.

Hyperparameter Search.ipynb - This Jupyter Notebook performs hyperparameter search on the CIFAR dataset to automatically acquire the best hyperparameters in a specific range.

CIFAR10 Train Test.ipynb - This Jupyter Notebook leverages the optimal searched hyperparameters to train the neural network on the entire CIFAR-10 dataset and compute the test accuracy. Besides, it includes the comparison between manually parameter performance and searched hyperparameters.

CIFAR100 Train Test.ipynb - This Jupyter Notebook leverages the optimal searched hyperparameters to train the neural network on the entire CIFAR-100 dataset and compute the test accuracy. Besides, it includes the comparison between manually parameter performance and searched hyperparameters.

You can easily access them from our Github Repo https://github.com/ecbme4040/e4040-2021fall-project-csgo, and keep clicking Run to run every cell in order. In case you encountered error, please RESTART THE KERNEL and try again from the beginning.

The cifar10 and cifar100 dataset is located in https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz and "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

Details about modules are listed by the commend inside the .py files, you can also check .pdf file in this repo, which contains pseudocode for all key modules in Section 4.2. If you have any other questions when running our project, please don't hesitate to let us know by emailing hl3515@columbia.edu or even call at (646) 906 -0624. Thank you!

# Organization of this directory
The directory tree is shown below.

![c257e67232afc2bbcd38c8cc52b0da2](https://user-images.githubusercontent.com/90977332/147037884-03a2a7e4-596b-4a33-8322-96773bf1280f.png)

Because the git push shows different local files so the tree can't be import from jupyter note book, but we give a screen shot here, please check!!!
