# TabPFN

We created a Colab notebook, that lets you interact with our scikit-learn interface at [https://colab.research.google.com/drive/1J0l1AtMVH1KQ7IRbgJje5hMhKHczH7-?usp=sharing](https://colab.research.google.com/drive/1J0l1AtMV_H1KQ7IRbgJje5hMhKHczH7-?usp=sharing)

We also created two demos. One to experiment with the TabPFNs predictions (https://huggingface.co/spaces/TabPFN/TabPFNPrediction) and one to check cross-
validation ROC AUC scores on new datasets (https://huggingface.co/spaces/TabPFN/TabPFNEvaluation). Both of them run on a weak CPU, thus it can require a little bit of time.
Both demos are based on a scikit-learn interface that makes using the TabPFN as easy as a scikit-learn SVM.

## Installation
```
conda create -n TabPFN python=3.7
$environment_path$/pip install -r requirements.txt
```

To run the autogluon baseline please create a separate environment and install autogluon==0.4.0, installation in the same environment as our other baselines is not possible.

## Getting started

A simple usage of our sklearn interface is:
```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
# When N_ensemble_configurations > #features * #classes, no further averaging is applied.

classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))
```

### TabPFN Usage

TabPFN is different from other methods you might know for tabular classification.
Here, we list some tips and tricks that might help you understand how to use it best.

- Do not preprocess inputs to TabPFN. TabPFN pre-processes inputs internally. It applies a z-score normalization (`x-train_x.mean()/train_x.std()`) per feature (fitted on the training set) and log-scales outliers [heuristically]. Finally, TabPFN  applies a PowerTransform to all features for every second ensemble member. Pre-processing is important for the TabPFN to make sure that the real-world dataset lies in the distribution of the synthetic datasets seen during training. So to get the best results, do not apply a PowerTransformation to the inputs.
- TabPFN expects scalar values only (you need to encode categoricals as integers e.g. with [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder)). It works best on data that does not contain any categorical or NaN data (see [Appendix B.1]).
- TabPFN ensembles multiple input encodings per default. It feeds different index rotations of the features and labels to the model per ensemble member. You can control the ensembling with `TabPFNClassifier(...,N_ensemble_configurations=?)`
- TabPFN does not use any statistics from the test set. That means predicting each test example one-by-one will yield the same result as feeding the whole test set together.
- TabPFN is differentiable in principle, only the pre-processing is not and relies on numpy.



