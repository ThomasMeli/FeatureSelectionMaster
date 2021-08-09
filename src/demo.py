"""
A 'live testing' framework to educate around feature importances
and also verify that this works while also showing the
sensitivity and specificity of the feature selection algorithms.


"""
from sklearn.datasets import make_regression

class RegressionProblem:
    def __init__(self,
                 n_informative,
                 n_noisy):
        pass

    def see_effect_of_noisy_features(self):
        """
        Gradually add noisy features
        and run through all estimators
        to see which ones are robust
        to noise and which ones aren't.

        """
        pass

    def see_effect_of_pointless_features(self):
        pass

    def see_effect_of_many_informative_features(self):
        """
        At what point does more informative features
        not matter?

        Diversity of information mattering more
        than just 'information'

        """
        pass

    def get_classification_report_of_feature_identification(self):
        """
        How well do classifiers find the positive features?
        What is sensitivity?
        What is specificity?

        """
        pass


    def find_feature_interactions(self):
        pass








