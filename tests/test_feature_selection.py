
from ..src.modules.opt_feat_sel import *
from ..src.modules.opt_cfg import CFG

from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.datasets import make_regression

from hypothesis import given, assume, note, settings, Verbosity, strategies as st
from hypothesis.extra.pandas import column, data_frames


################

test_regression_target_col_name = "target"

################

class ModelManager:
    pass

def initialize_test_models():
    pass


def initialize_test_feat_manager(models = None,
                                 target_type=None):

    models, target_type = get_adaboost_model(models, target_type)

    return FeatureImportanceManager(models = models,
                                    target_type = target_type)


def get_adaboost_model(models, target_type):
    if target_type is None:
        target_type = "regression"
    if models is None:
        if target_type == "regression":
            models = [AdaBoostRegressor()]
        elif target_type == "classification":
            models = [AdaBoostClassifier()]

    return models, target_type


def initialize_regression_df():
    X, y = make_regression()

    df = pd.DataFrame(X)
    df[test_regression_target_col_name] = y

    return df

############### test defaults ################

t_fm = initialize_test_feat_manager()
test_regression_df = initialize_regression_df()

##############################################

def test_featureimportancemanager():
    pass

def test_fmanager_models_have_fit_method():
    print(t_fm)

def test_feature_name_persistence():
    pass

def test_models_in_manager_is_list():
    pass

def test_regressions_work_with_feature_manager():
    pass

# Like a mini integration test
def test_feature_manager_run_all():
    t_fm.fit(test_regression_df, "target")
    t_fm.run_all()

def test_feature_manager_splits_Xy_correctly():

    t_fm.fit(test_regression_df, "target")
    print(t_fm.X.shape)
    print(t_fm.y.shape)

def test_feat_manager_remove_low_variance_features():
    # Use Hypothesis for this.

    t_fm.fit(test_regression_df, "target")
    t_fm.do_variance_threshold()

    t_fm.do_variance_threshold(threshold=0.8)


    pass

def test_feat_manager_permutation_importance_returns_results():
    pass

def test_model_with_and_without_features():
    pass

########## Test class for common buildup and takedown?######
# Permutation importance for example.

def test_find_intrinsic_dimensionality():
    t_fm.fit(test_regression_df, "target")
    t_fm.find_intrinsic_dimensionality()

def test_permutation_importance():
    t_fm.fit(test_regression_df, "target")
    t_fm.do_permutation_importance()

def test_permutation_report_created():

    # simpler, see if filename exists
    # before and after run.

    # should be in the log dir

    # assert directory content is not the same
    # before and after the run.

    pass

def test_columns_match():
    pass

def test_column_matcher():
    pass

def get_fitted_model_importances():
    pass

# Stress test the graphs - How to make assertions for this?
# Or can this happen after
# @given(stress_test_df = data_frames(
# [
#         column('var1', elements=st.floats()),
#         column('var2', elements=st.floats()),
#         column('var3', elements=st.floats()),
#         column('target', elements=st.floats())
#     ]
# ))
# @settings(max_examples=100, verbosity=Verbosity.verbose)
# def test_stress_premutation_importance_graphs(stress_test_df):
#     t_fm.fit(stress_test_df, "target")
#
#     t_fm.do_permutation_importance(show_plots=True)
#
#     # assert

def test_variance_thresh_column_names_are_preserved():
    t_fm.fit(test_regression_df, "target")
    varthreshed_X = t_fm.do_variance_threshold(threshold=.90)

    initial_cols = set(list(t_fm.colnames_initial))
    final_cols = set(list(t_fm.varthreshed_colnames))

    assert final_cols.issubset(initial_cols)


# create given for creating strange colnames.
def test_stress_variance_thresh_columns_are_preserved():
    pass


def test_log_dir_will_not_overwrite_old_logs():
    # Create an overwrite = False default for log

    # find log names in folder based on splitting on _

    # or make sure the exact file will not be made

    # anticipate future file

    # assert this_log_name is not in old_log_names
    pass


def test_overall_report_created():
    t_fm.fit(test_regression_df, "target")
    run_name = t_fm._return_run_name(t_fm.run_id)

    # look for file in directory that starts with runname.
    #runname_overall_report.csv
    pass

def test_overall_report_records_variance():

    # assert that the order of the index is the same
    # assert that everything matches on index.
    # assert that n_cols is the same.
    pass

def test_numerical_report_records_variance():

    # assert that the order of the index is the same
    # assert that everything matches on index.
    # assert that n_cols is the same.


    pass

def test_overall_report_records_permutation():
    pass

def test_corr_handles_no_features_above_thresh():
    pass

def test_get_highly_correlated_feature_name_pairs():
    t_fm.fit(test_regression_df, "target")
    t_fm.get_highly_correlated_feature_name_pairs(
        threshold=0.5
    )

    print(t_fm.X_corr_pairs)
    print(len(t_fm.X_corr_pairs))

def test_setting_of_model():
    # AttributeError: 'RandomForestRegressor' object has no attribute 'estimators_'

    # model = self.model
    # AttributeError: 'FeatureImportanceManager' object has no attribute 'model'

    pass

def test_return_safe_savename():

    # Do the same run multiple times.
    # Should save unique runs in folder with a, b, c, etc.

    pass

def do_test_do_tree_importances_is_deterministic():
    # Should get the 'same' results on each run if using random_seed.

    pass

def test_return_rank_with_feature():

    pass

def test_three_correlations():
    """
    Spearman, pearson, etc.
    Extract "rank_finder"
    :return:

    """
    pass

def test_do_tree_importances():
    t_fm.fit(test_regression_df, "target")
    t_fm.do_tree_importances()

def test_do_tree_importances_rank():
    pass

def test_set_n_feats_by_intrinsic_dimensionality():
    pass

def test_model_has_feature_importance_attr():
    pass

# Mark as implementation.
def test_StatReduce_construction():
    newvarthresh = StatManager(test_regression_df)
    print(newvarthresh)
