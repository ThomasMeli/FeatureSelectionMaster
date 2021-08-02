
class IntrinsicDimensionality:
    """
    Performs feature selection and analysis through dimensionality reduction.
    """

    def fit(self):
        pass


    def set_n_feats_by_intrinsic_dimensionality(self,
                                                goal_explained_variance = 0.95,
                                                check_if_excluded_features_correlate_or_have_interactions = False):
        """
        Gets the number of features where goal_explained_variance is reached.

        Double checks to see if any remaining features
        :param goal_explained_variance:
        :return:
        """
        pass

    def find_intrinsic_dimensionality(self, aimfor_cumulative_explained_variance = 0.95):
        self.aimfor_cumulative_explained_variance = aimfor_cumulative_explained_variance
        self._find_intrinsic_dimensionality()

    def _find_intrinsic_dimensionality(self):
        n_half_features = int(self.X_nfeats_initial / 2 )
        n_two_thirds_features = int(2 * self.X_nfeats_initial / 3 )

        self._do_linear_pcas(n_half_features)
        self._do_linear_pcas(n_two_thirds_features)

    def _do_linear_pcas(self, n_feats, whiten_compare = False):

        if self.verbose:
            print(f"\nRunning Linear PCA with {n_feats} Features")

        linear_pca = PCA(n_components=n_feats)
        linear_pca.fit(self.X)

        if whiten_compare:
            linear_pca_whiten = PCA(n_components=n_feats, whiten=True)
            linear_pca_whiten.fit(self.X)

        if self.verbose:
            half_feats_explained_variance = round(sum(linear_pca.explained_variance_ratio_), 2)
            print(f"First few features have explained variance {np.round(linear_pca.explained_variance_ratio_[:5], 2)}")
            print(f"{n_feats} Features Explained Variance: {half_feats_explained_variance}")

            if whiten_compare:
                half_feats_whiten_explained_variance = round(sum(linear_pca_whiten.explained_variance_ratio_), 2)
                print(f"{n_feats} Features Explained Variance Whitened: {half_feats_whiten_explained_variance}")
