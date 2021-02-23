
# Feature Importance in Machine Learning

When developing predictive models and risk measures, it's often helpful to know which features are making the most difference. This is easy to determine in simpler models such as linear models and decision trees. However as we move to more complex models to achieve high performance, we usually sacrifice some interpretability. In this assignment we'll try to regain some of that interpretability using Shapley values, a technique which has gained popularity in recent years, but which is based on classic results in cooperative game theory. 


## Permuation Method for Feature Importance:

In the permutation method, the importance of feature  i  would be the regular performance of the model minus the performance with the values for feature  i  permuted in the dataset. This way we can assess how well a model without that feature would do without having to train a new model for each feature.

## Shapley Values for Random Forests:
use Shapley values to try and understand the model output on specific individuals. In general Shapley values take exponential time to compute, but luckily there are faster approximations for forests in particular that run in polynomial time., we might also want to understand model output in aggregate. Shapley values allow us to do this as well.

#### Visualizing Interactions between Features
The shap library also lets you visualize interactions between features using dependence plots. These plot the Shapley value for a given feature for each data point, and color the points in using the value for another feature. This lets us begin to explain the variation in shapley value for a single value of the main feature.
 
 
- pip install lifelines shap
#### to train:
- run python train/training.py
#### to interpret :
- run in cmd prmt :   python interpreter.py
- visulaization will be stored in visulaization/shap folder 
