# from models import LinearModels, TreeModels, DenseBaseline
# from statsmodels.stats.outliers_influence import variance_inflation_factor

class OptimizerManager:
    pass



class OptimizerFullStrategy:
    """

        Creates several baselines.

        0. Setup
        -- Configure a split strategy.
        -- validate data is ML ready (NaNs, infs, duplicates, etc.)

        1. Unmodified dataset with linear model, tree model, nn model.

        # TRAIN LOOP: (linear, tree, Dense NN, Noise, and Random_seed)
        # Baseline of Linear
        # Baseline of Tree
        # (later) Baseline of Dense NN
        # Baseline Noise Robustness (Add some noisy features - measure baseline robustness to noise).
        # Baseline robustness to random_seed.

        2. Variance Thresholded Model.
        # Variance threshold caused ____ change in score.

        3. Remove multi-colinear variables
        # Initial VIF Score.
        # split into two groups.
        # Validate lack of multicolinearity in each group.
        # Run with one group. - A baseline... VIF Score
        # Run with the other. - B baseline... VIF Score.
        # Removing Multi-Colinear Features caused ____ change in score in group A, and ___ in group B.

        4. Validation with noisy features.
        - test models being used are robust against noisy uninformative features
        - and use these uninformative features as a 'feature-information' baseline.
        - Adding ____ noisy features caused a ____ change in score.
        - This means ___ amount of improvement may be due to random factors.

        3. Permutation Importance (w/ threshold) (w/ negative feature interactions)
        - export results and rank positive permutation importances.
        - create feature interactions with negatives.
        # Taking only positive permutation importance features (in linear models) caused ___ change.

        ### Polynomial transform LOOP - create feature interactions until no negative perm features?
        # Do polynomial transform on negative features. run with linear......
        # ______

        3. PCA with 95% variance on linear models.  Find n-dim.


        4. Tree based feature selection.

        5. Run predictions_analyzer cross_corr to find most diverse and good predictions.

        6.

        """

    def training_loop(self,
                      do_linear = True,
                      do_trees = True,
                      do_nn = False,
                      do_noisy_feature = True,
                      do_random_seed_variance = True):
        pass

