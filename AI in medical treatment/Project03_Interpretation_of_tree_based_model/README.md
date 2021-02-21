
# Feature Importance in Machine Learning

When developing predictive models and risk measures, it's often helpful to know which features are making the most difference. This is easy to determine in simpler models such as linear models and decision trees. However as we move to more complex models to achieve high performance, we usually sacrifice some interpretability. In this assignment we'll try to regain some of that interpretability using Shapley values, a technique which has gained popularity in recent years, but which is based on classic results in cooperative game theory. 


## Permuation Method for Feature Importance:

In the permutation method, the importance of feature  i  would be the regular performance of the model minus the performance with the values for feature  i  permuted in the dataset. This way we can assess how well a model without that feature would do without having to train a new model for each feature.