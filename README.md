# Machine Learning Engineer Nanodegree
# Model Evaluation and Validation
## Project: Predicting Boston Housing Prices

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [Scikit-learn](http://scikit-learn.org/stable/)
- [AudioFeaturizer](https://pypi.org/project/AudioFeaturizer/)
- [Librosa](https://pypi.org/project/librosa/)
- [Statistics](https://pypi.org/project/statistics/)
- [Seaborn](https://pypi.org/project/seaborn/)


You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included. 

### Code

Code is provided in the `AudioGenreClassifier.ipynb` notebook file.

### Run

In a terminal or command window, navigate to the top-level project directory `Music-Genre-Classification/` (that contains this README) and run one of the following commands:

```bash
ipython notebook AudioGenreClassifier.ipynb
```  
or
```bash
jupyter notebook AudioGenreClassifier.ipynb
```
or open with Juoyter Lab
```bash
jupyter lab
```

This will open the Jupyter Notebook software and project file in your browser.

### Data

The dataset GTZAN used for building music genre classification has been taken from Kaggle . It consists of 1000 samples of songs. [GTZAN Dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification).

**Features**

1. Mel Frequency Cepstral Coefficients(mean and variance)
2. Spectral bandwidth(mean and variance)
3. Spectral centroid(mean and variance)
4. Zero crossing rate(mean and variance)
5. Spectral roll-off(mean and variance)
6. Beats location (mean and variance)
7. Estimated global tempo(mean and variance)
8. Chroma short time fourier transform (mean and variance)
9. Root mean square energy for each frame(mean and variance)
10. Harmonic component(mean and variance) 
11. Percussive component(mean and variance)

**Exploratory Data Analysis**

1. No null values were present in the dataset
2. The target variable is the music genre which is categorical in nature. Hence
label encoding was performed to obtain better results.
3. We analysed that the ranges of features differ by a significant amount. So in
order to avoid the algorithm to give more importance to the feature with
higher range, the feature set was standardised.
4. Relevant features were selected using the insights drawn from variance and
correlation. A feature with low variance doesn't have much predictive power and doesn't contribute much. So features like harmonic_var(variance=1.357712e-04),percussive mean (variance=1.170194e-06), percussive var(variance=4.226615e-05) and harmonic mean(variance=2.835795e-06) were removed. If two features are highly correlated then the feature with highest correlation with the target label was kept in relevant features and the other with lower variance with the target variable was removed.
5. We observed that there is a significant difference between mean, 75th percentile and maximum value for several features thus indicating presence of outliers. We removed the outliers using percentile capping method where anything greater than 99th percentile and lesser than 1st percentile has been removed. After removing the outliers and important feature selection the final dataset contains : 988 training examples and 56 feature sets.

**Design Choices**

● So we started with SVM as our baseline and then we looked for other classifiers Which includes KNN, NN, LDA, QDA, Naive Bayes and Logistic regression.
● We used accuracy as a evaluation metric to analyse the performance of different model.
● We have used Grid Search method along with k- fold cross validation resampling technique to choose the best parameter from given set of parameters. The chosen parameters gives best performance on the dev set and hence these parameters were used to finally estimate the genre of test data.

**Result**

1. `Poly Kernel SVM`: 73.4%
2. `KNN`: 72%
3. `Random Forest`: 72%
4. `RBF Kernel SVM`: 72%
5. `Neural Network`: 71%
6. `Linear Kernel SVM`: 71%
7. `Latent Dirichlet allocation`: 71%
8. `Sigmoid Kernel SVM`: 71%



