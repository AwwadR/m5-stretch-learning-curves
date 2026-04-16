## Analysis

The learning curve shows that the model is suffering more from high bias than high variance. The training and validation F1 scores get closer together as the training size increases, and both curves stop at a moderate score. This means the model is not overfitting badly, but it is also not learning enough to get stronger results.

I improved the code by removing customer_id and by using a preprocessing pipeline with scaling for numeric columns and one hot encoding for categorical columns. I also used StratifiedKFold and F1 score because the dataset is imbalanced, so this gives a more reliable evaluation. More data may help a little, but probably not a lot because the validation curve becomes almost flat. My next step would be to improve the features, tune logistic regression more, or try a more flexible model.
