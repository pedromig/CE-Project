import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import collections


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

    def __call__(self: object):
        return self._mean, self._median, self._mode, self._std, self._var

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

    def ttest(self: object, a, b, *args,
              paired=True, equal_var=True, **kwargs):
        if not paired:
            return st.ttest_ind(self._data[a], self._data[b],
                                equal_var=equal_var, *args, **kwargs)
        return st.ttest_rel(self._data[a], self._data[b],
                            *args, **kwargs)

    def one_way_anova(self: object, paired=False, *args, **kwargs):
        if not paired:
            return st.f_oneway(*(c for _, c in self._data.iteritems()))

        grand_mean = self._data.values.mean()

        row_means = self._data.mean(axis=1)
        column_means = self._data.mean(axis=0)

        n, k = len(self._data.axes[0]), len(self._data.axes[1])
        N = self._data.size

        # degrees of freedom
        df_total = N - 1
        df_between = k - 1
        df_subject = n - 1
        df_within = df_total - df_between
        df_error = df_within - df_subject

        # compute variances
        ss_between = sum(n*[(m - grand_mean)**2 for m in column_means])
        ss_within = sum(sum([(self._data[col] - column_means[i])
                             ** 2 for i, col in enumerate(self._data)]))
        ss_subject = sum(k * [(m - grand_mean)**2 for m in row_means])
        ss_error = ss_within - ss_subject

        # Compute Averages
        ms_between = ss_between / df_between
        ms_error = ss_error / df_error

        fstat = ms_between / ms_error
        p_value = st.f.sf(fstat, df_between, df_error)

        AnovaResult = collections.namedtuple(
            "F_onewayResult", ["statistic", "pvalue"])
        return AnovaResult(fstat, p_value)

    def ttest_effect_size(statistic: float, df: int):
        return np.sqrt(statistic**2 / (statistic**2 + df))

    def mann_whitney_effect_size(statistic: float, obs1: int, obs2: int):
        nobs = obs1 + obs2
        mean = obs1 * obs2 / 2
        std = np.sqrt(obs1 * obs2 * (obs1 + obs2 + 1) / 12)
        z_score = (statistic - mean) / std
        return z_score / np.sqrt(nobs)

    def wilcoxon_effect_size(self, stat: float, sample: int, nobs: int):
        mean = sample * (sample + 1) / 4
        std = np.sqrt(sample * (sample + 1) * (2 * sample + 1) / 24)
        z_score = (stat - mean) / std
        return z_score/np.sqrt(nobs)

    def mann_whitney(self: object, a, b, *args, **kwargs):
        return st.mannwhitneyu(self._data[a], self._data[b], *args, **kwargs)

    def wilcoxon(self: object, a, b, *args, **kwargs):
        return st.wilcoxon(self._data[a], self._data[b], *args, **kwargs)

    def kruskal_wallis(self: object, *args, **kwargs):
        return st.kruskal(*(c for _, c in self._data.iteritems()),
                          *args, **kwargs)

    def friedman_chi(self: object, *args, **kwargs):
        return st.friedmanchisquare(*(c for _, c in self._data.iteritems()),
                                    *args, **kwargs)

    def boxplot(self: object, title="Box Plot"):
        sns.swarmplot(data=self._data, color=".25")
        sns.boxplot(data=self._data)
        plt.title(title)
        # plt.show()

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
    file = "step-stats.csv"
    rand = Statistics(file, header=None, usecols=[0, 1], names=[0, 1])
    elite = Statistics(file, header=None, usecols=[0, 2], names=[0, 1])

    # rand.describe(plot=True)
    # elite.describe(plot=True)

    # rand.histogram(normal=True)
    # elite.histogram(normal=True)

    print(rand.kstest())
    print(elite.kstest())

    rw = rand.wilcoxon(0, 1)
    ew = elite.wilcoxon(0, 1)
    print(rw)
    print(ew)

    x = np.count_nonzero(rand._data[0] - rand._data[1]) - 1
    print(elite.wilcoxon_effect_size(rw.statistic, x, len(rand._data)))

    y = np.count_nonzero(elite._data[0] - elite._data[1]) - 1
    print(elite.wilcoxon_effect_size(ew.statistic, y, len(elite._data)))
