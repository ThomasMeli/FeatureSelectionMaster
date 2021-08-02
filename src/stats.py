

class StatManager:
    """
    Performs feature selection and analysis through statistical analysis.
    """

    def __init__(self,
                 X,
                 verbose = False,
                 variance_threshold = 0.95):

        self.variance_threshold = variance_threshold
        self.X = X
        self.verbose = verbose

    def _do_variance_threshold(self):
        varthresh = VarianceThreshold(self.variance_threshold)
        varthresh.fit(self.X)

        self.varthresh_support_mask = varthresh.get_support()
        self.varthreshed_colnames = \
                self.X.columns[self.varthresh_support_mask]


        varthreshed_X = pd.DataFrame(
            varthresh.transform(self.X),
            columns = self.varthreshed_colnames)

        self.X_nfeats_after_varthresh = varthreshed_X.shape[1]

        # TODO: DO not save data, save columns and make a getter to GET the data.
        self.varthreshed_X = varthreshed_X
        # self.varthreshed_X = self._get_varthreshed_X()

        if self.verbose:
            self._report_variance_threshold()

    def _report_variance_threshold(self):
        print("\n")
        print(f"With Threshold: {self.variance_threshold}")
        print(f"Before Variance Threshold: {self.X_nfeats_initial} \nAfter Variance Threshold {self.X_nfeats_after_varthresh}")
        print("Data Accessible through obj.varthreshed_X")

    def _get_varthreshed_X(self):
        pass

    def get_X_correlation(self):
        pass

    def get_Xy_corelation(self):
        pass

    def get_correlation_feature_pairs(self):
        pass

    def compare_thresholded_data_with_models:
        """
        Runs one model on data with and without threshold
        to see which performs better and by how much.

        :return:
        """
        pass


    def _report_variance_threshold(self):
        print("\n")
        print(f"With Threshold: {self.variance_threshold}")
        print(f"Before Variance Threshold: {self.X_nfeats_initial} \nAfter Variance Threshold {self.X_nfeats_after_varthresh}")
        print("Data Accessible through obj.varthreshed_X")

    def _get_varthreshed_X(self):
        pass


    def get_highly_correlated_feature_name_pairs(self,
                                                 threshold = 0.7):
        """
        In process - right now this function gets ALL correlated features and does
        some intelligent processing with it to enhance interpretation.
        Todo:  filter around threshold or top_n.

        Returns a set of tuples of strings of highly correlated feature pairs.

        :return:
        """

        self.X_corr = self.X.corr()
        self.X_corr = self.X_corr.where(
            np.triu(np.ones(self.X_corr.shape), k=1).astype(bool)
        )

        self.X_corr.to_csv(self.base_save_name + "_X_corr.csv")

        self.X_corr_pairs = self.X_corr \
            .unstack() \
            .sort_values(kind="quicksort") \
            .dropna()

        self.X_corr_pairs = self.X_corr_pairs[self.X_corr_pairs != 1]
        # name axis meaningfully.

        self.X_corr_pairs.to_csv(self.base_save_name + "_X_corr_pairs.csv", index=True)

        self.X_corr_above_thresh = self.X_corr[abs(self.X_corr) > threshold] \
            .dropna(how="all")

        # todo: Delete this.  Reset_index is meaningless in this Series.
        self.X_corr_pairs.reset_index()
        print("after reset index")

        print(self.X_corr_pairs.head())
        self.X_corr_pairs_set = set(self.X_corr_pairs.index.to_list())
        print(len(self.X_corr_pairs_set))


    def get_highly_correlated_feature_names_keep_best(self,
                                                      df,
                                                      target_col):
        """
        Keeps ONE of the highest correlated pairs.
        It will decide this based on which element is more correlated with
        the target variable.

        :return:
        """
        pass

    def get_highly_correlated_feature_names(self):
        """
        Returns ALL of the highly correlated features.
        This is just for information only and will make models worse
        if you get rid of all of these.

        Use get_highly_correlated_feature_names_keep_best instead.

        :return:
        """