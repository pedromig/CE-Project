import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


class Statistics:

    # Set styling for statistic plots
    sns.set()

    def __init__(self: object, file: str, *args, **kwargs) -> None:
        self._data = pd.read_csv(file, *args, **kwargs)

        # Basic Statistics
        self.min, self.max = self._data.min(axis=0), self._data.max(axis=0)
        self.mean = self._data.mean(axis=0)
        self.median = self._data.median(axis=0)
        self.mode = self._data.mode(axis=0)
        self.std = self._data.std(axis=0)
        self.var = self._data.var(axis=0)
        self.skewness = self._data.skew(axis=0)
        self.kurtosis = self._data.kurtosis(axis=0)
        self.q25 = self._data.quantile(.25)
        self.q50 = self._data.quantile(.50)
        self.q75 = self._data.quantile(.75)

    def levene(self: object):
        """
        Inferential statistic used to assess the equality of
        variances for a variable calculated for two or more groups.
        """
        return st.levene(*(c for _, c in self._data.iteritems()))

    def kstest(self: object, dist='norm', *args, plot=False, **kwargs):
        """
        The Kolmogorov-Smirnov test is a hypothesis test procedure
        for determining if two samples of data are from the same distribution.
        This implementation acesses the goodness of fit between the user
        provided distribuition and the normal distribuition.
        """
        results = []
        for k, c in self._data.iteritems():
            norm = (c - c.mean()) / (c.std() / np.sqrt(len(c)))
            results.append(st.kstest(norm, dist, *args, **kwargs))
        if plot:
            self.histogram(normal=True)
        return results

    def anderson(self: object, *args, dist='norm', plot=False, **kwargs):
        """
        Anderson-Darling to test if a sample of data came from a
        population with a specific distribution. It is a modification
        of the Kolmogorov-Smirnov (K-S) test and gives more weight
        to the tails than does the K-S test.

        If the returned statistic is greater then the critical value for
        a significance level swe can reject the null hypothesis that states
        that the data came from a a distribution of the type described.
        """
        results = []
        for k, c in self._data.iteritems():
            norm = (c - c.mean()) / (c.std() / np.sqrt(len(c)))
            results.append(st.anderson(norm, dist, *args, **kwargs))
        if plot:
            self.histogram(normal=True)
        return results

    def shapiro(self: object, *args, plot=False, **kwargs):
        """
        Shapiro–Wilk statistic for testing if the population
        is normally distributed.
        """
        results = []
        for _, c in self._data.iteritems():
            norm = (c - c.mean()) / (c.std() / np.sqrt(len(c)))
            results.append(st.shapiro(norm, *args, **kwargs))
        if plot:
            self.histogram(normal=True)
        return results

    def ttest(self, a, b, independent=True, *args, **kwargs):
        return st.ttest_ind(self._data[a, :], self._data[b, :],
                            equal_var=independent) \
            if independent else st.ttest_rel(a, b, *args, **kwargs)

    # Non Parametric
    def mann_whitney(self: object, a, b, *args, **kwargs):
        return st.mannwhitneyu(a, b, *args, **kwargs)

    def wilcoxon(self: object, a, b, *args, **kwargs):
        return st.wilcoxon(a, b)

    def kruskal_wallis(self: object, a, b, *args, **kwargs):
        return st.kruskal(*self._data)

    def friedman_chi(self: object):
        return st.friedmanchisquare(*self._data)

    def one_way_ind_anova(self: object, independent=True, *args, **kwargs):
        if independent:
            return st.f_oneway(*self._data)

        grand_mean = self._data.values.mean()
        # grand_variance = data_frame.values.var(ddof=1)

        row_means = self._data.mean(axis=1)
        column_means = self._data.mean(axis=0)

        # n = number of subjects; k = number of conditions/treatments
        n, k = len(self._data.axes[0]), len(self._data.axes[1])
        # total number of measurements
        N = self._data.size  # or n * k

        # degrees of freedom
        df_total = N - 1
        df_between = k - 1
        df_subject = n - 1
        df_within = df_total - df_between
        df_error = df_within - df_subject

        # compute variances
        SS_between = sum(n*[(m - grand_mean)**2 for m in column_means])
        SS_within = sum(sum([(self._data[col] - column_means[i])
                             ** 2 for i, col in enumerate(self._data)]))
        SS_subject = sum(k * [(m - grand_mean)**2 for m in row_means])
        SS_error = SS_within - SS_subject
        # SS_total = SS_between + SS_within

        # Compute Averages
        MS_between = SS_between / df_between
        MS_error = SS_error / df_error
        # MS_subject = SS_subject / df_subject

        # F Statistics
        F = MS_between / MS_error
        # p-value
        p_value = st.f.sf(F, df_between, df_error)

        return (F, p_value)

    # Effect size
    def effect_size_t(stat, df):
        r = np.sqrt(stat**2/(stat**2 + df))
        return r

    def effect_size_mw(stat, n1, n2):
        """
        n_ob: number of observations
        """
        n_ob = n1 + n2
        mean = n1*n2/2
        std = np.sqrt(n1*n2*(n1+n2+1)/12)
        z_score = (stat - mean)/std
        print(z_score)
        return z_score/np.sqrt(n_ob)

    def effect_size_wx(stat, n, n_ob):
        """
            n: size of effective sample (zero differences are excluded!)
            n_ob: number of observations
            """
        mean = n*(n+1)/4
        std = np.sqrt(n*(n+1)*(2*n+1)/24)
        z_score = (stat - mean)/std
        return z_score/np.sqrt(n_ob)

    def boxplot(self: object, title="Box Plot"):
        sns.swarmplot(data=self._data, color=".25")
        sns.boxplot(data=self._data)
        plt.title(title)
        plt.show()

    def histogram(self: object, title="Histogram", bins=25, normal=False):
        for c in self._data.columns:
            sns.histplot(data=self._data, x=c, bins=bins,
                         kde=normal, alpha=0.6)
            plt.title(title)
            plt.show()

    def describe(self: object, plot=False):
        print(self.__str__())
        if plot:
            self.boxplot()

    def __str__(self: object):
        return f"""\
            \rMin: {" | ".join(str(i) for i in self.min)}
            \rMax: {" | ".join(str(i) for i in self.max)}
            \rMean: {" | ".join(str(i) for i in self.mean)}
            \rMedian: {" | ".join(str(i) for i in self.median)}
            \rMode: {" | ".join(str(self.mode[i][0])
                            for i in range(self.mode.shape[1]))}
            \rVar: {" | ".join(str(i) for i in self.var)}
            \rStd: {" | ".join(str(i) for i in self.std)}
            \rSkew: {" | ".join(str(i) for i in self.skewness)}
            \rKurtosis: {" | ".join(str(i) for i in self.kurtosis)}
            \rQ25: {" | ".join(str(i) for i in self.q25)}
            \rQ50: {" | ".join(str(i) for i in self.q50)}
            \rQ75: {" | ".join(str(i) for i in self.q75)}\
            """


if __name__ == '__main__':
    filename_1 = 'pulse_rate.txt'
    stats = Statistics("sphere.txt", header=None, sep="\t")
    stats.describe(plot=True)
    print(stats.levene())
    for i in stats.kstest():
        print(i)
    for i in stats.shapiro():
        print(i)
    for i in stats.anderson():
        print(i)
