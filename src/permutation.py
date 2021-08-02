
class PermutationManager:
    def __init__(self):
        pass


    def _report_permutation_importance(self,
                                       this_model_name,
                                       show_sample_outputs = True,
                                       show_plots = False,
                                       show_recommendations = True):

        ## Extract to Report and Report Manager + Persistence Manager

        self.perm_report_df = pd.DataFrame(index=self.colnames_initial)

        perm_means = self.permutation_importance_results.importances_mean
        perm_std = self.permutation_importance_results.importances_std

        this_mean_name = str('mean_importance_' + this_model_name)
        this_std_name = str('std_importance_' + this_model_name)

        self.perm_report_df[this_mean_name] = perm_means
        self.perm_report_df[this_std_name] = perm_std

        self.perm_report_df.sort_values(by=this_mean_name,
                                        ascending=False,
                                        inplace=True)

        csv_name = self.run_id + "_permutation_report.csv"
        self.perm_report_df.to_csv(self.log_dir + csv_name)

        ## Extract to Plots and Report Manager and Persistence Manager

        if show_plots:
            self._fit_permutation_graphs(this_mean_name, this_std_name)

        if show_recommendations:
            self._recommendations_of_permutation_importance()

    def _fit_permutation_graphs(self, this_mean_name, this_std_name):
        self.ax_perm_mean = sns.barplot(x=self.perm_report_df.T.columns,
                            y=self.perm_report_df[this_mean_name],
                            order=self.perm_report_df.T.columns)

        self.ax_perm_std = sns.barplot(x=self.perm_report_df.T.columns,
                            y=self.perm_report_df[this_std_name],
                            order=self.perm_report_df.T.columns)

        print("plots available at this_obj.show_permutation_graphs()")


    def show_permutation_graphs(self):

        self.ax_perm_mean.show()
        self.ax_perm_std.show()

        plt.show()

    def _recommendations_of_permutation_importance(self,
                                                   std_thresh = 0.05,
                                                   mean_thresh = 0.05,
                                                   top_n = None):

        if top_n is None:
            top_n = self.n_top_feats

        print("\n")
        print(self.perm_report_df.head())
        print(self.perm_report_df.tail())

        mean_cols_df = self.perm_report_df.loc[:, self.perm_report_df.columns.str.startswith('mean')]
        std_cols_df = self.perm_report_df.loc[:, self.perm_report_df.columns.str.startswith('std')]

        above_mean_thresh = mean_cols_df[mean_cols_df > mean_thresh].dropna(axis=0)
        negative_means = mean_cols_df[mean_cols_df < 0].dropna(axis=0)
        below_std_thresh = std_cols_df[std_cols_df < std_thresh].dropna(axis=0)

        print("\n")
        print(f"Top n features")
        print(list(mean_cols_df.T.columns[:top_n]))
        print("\n")

        print(f"Features with possible interactions (negative permutation importance)")
        print("Features at the end of the list have the highest potential interaction value")

        print(list(negative_means.T.columns))
        print("\n")

        print(f"Columns above the mean threshold of {mean_thresh}:\n {above_mean_thresh}")
        print("\n")
        print(f"Columns above the std threshold of {std_thresh}:\n {below_std_thresh}")


