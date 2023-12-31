#Michael Grant
#Operating System: Windows 10/11
#Packages/Libraries: ggplot2, dplyr, ggpubr, dgof
#Data Files: "input_data/RunnerData.csv"

library(ggplot2)
library(ggpubr)
library(dplyr)
library(dgof)

setwd("C:/Users/Grant/OneDrive/Homework/Fall 2023/BIOL672/Assignment1")

#I will be using a Poisson distribution using the function rpois(n, lambda)
#Where n is the number of data points in the distribution and lambda is the
#expected value for the event occurring over a period of time.

datapoints <- 5000 #satisfies n
lambda <- 25.0 #satisfies lambda

rand_poissons <- rpois(n = datapoints, lambda = lambda)


#Return the sample mean and sample standard deviation
distribution_mean = mean(rand_poissons)
distribution_std = sd(rand_poissons)

#Output the results to the console
#sprintf is a function that can output strings that contain variables
sprintf("The sample mean is: %f", distribution_mean)
sprintf("The sample standard deviation is %f", distribution_std)

#Create a normal distribution from the mean and standard deviation of the Poisson
#distribution. This will be used for the bell curve overlay.
norm_distribution <- rnorm(5000, mean = distribution_mean, sd = distribution_std)

df <- data.frame(Normal = c(norm_distribution),
                 Poisson = c(rand_poissons))

#Create histogram for the Poisson distribution with density and normal distribution
histogram_poisson_distribution = ggplot() + 
  geom_histogram(df, mapping = aes(x=Poisson, y = ..density.., color= 'Histogram'), bins=35) + 
  geom_density(df, mapping = aes(x=Poisson, color='Histogram Density'), size=2) +
  geom_density(df, mapping = aes(x=Normal, color="Normal Density"), size=2) +
  scale_color_manual(values = c('black', 'red', 'blue')) +
  labs(x="Iterations until Event Occurs", y="Probability", colour = "Plots") +
  guides(color = guide_legend(override.aes = list(fill = c('black', 'red', 'blue')))) +
  theme(panel.background = element_blank(),
        panel.border = element_rect(colour="black", fill=NA, size=2),
        axis.ticks.length = unit(-0.25, 'cm'),
        legend.position = c(0.85, 0.88),
        legend.title.align = 0.5)

#Redirect the printed output to a file rather than the console
output <- character(0)
output <- c(output, sprintf("Sample Mean: %f", distribution_mean))
output <- c(output, sprintf("Sample Standard Deviation: %f", distribution_std))

invisible(sink("RunnerBMIResults/desc.txt"))

cat(output, sep="\n")

sink()

#Create a PDF of the generated plot 
ggsave("RunnerBMIResults/histo.pdf", plot=histogram_poisson_distribution)

#Load in the csv for the One-way ANOVA - I will be testing three groups of runners
#data. Each subgroup will be within a specific BMI range, and the null hypothesis
#is that by randomly selecting from each group, the average 5K time should remain
#the same.

#Reads in csv file and saves as a dataframe
runners_df <- read.csv("input_data/RunnerData.csv")

#Creates new dataframe containing only the columns we are concerned with and then
#deletes all rows that have values of "NA"
runners_df_clean <- na.omit(runners_df[, c("age", "bmi", "k5_ti_adj")])

#Creates a new column that categorizes each runner based on BMI.
#quantile takes a vector and paritions it into relatively even subdivisons.
breaks <- quantile(runners_df_clean$bmi, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE)
runners_df_clean$bmi_category <- cut(runners_df_clean$bmi, breaks = breaks, labels = c(1, 2, 3))

#Takes the dataframe and randomly selects 100 samples from each of the 3 categories.
#Take the current dataframe, groups it be a certain category, selects our sample size
#and then stops attempting to group dataframes
runners_anova_df <- runners_df_clean %>%
  group_by(bmi_category) %>%
  sample_n(size = 100, replace = TRUE) %>%
  ungroup()

#Eliminate any residual NAs that were formed
runners_anova_df <- na.omit(runners_anova_df)

#Conduct the oneway anova (aov() is another function that would work here as well)
anova_test <- oneway.test(runners_anova_df$k5_ti_adj ~ runners_anova_df$bmi_category)

#Print the resutls to text file
sink('RunnerBMIResults/ANOVA_RESULTS.txt')
print(anova_test)
sink()

#Obtain the mean and standard error of the mean for each category for use in an
#error bar plot.
df_group_1 <- runners_anova_df[runners_anova_df$bmi_category == 1,]
BMI_group_1_mean <- mean(df_group_1$k5_ti_adj)
BMI_group_1_error <- sd(df_group_1$k5_ti_adj) /
  sqrt(length(df_group_1))

df_group_2 <- runners_anova_df[runners_anova_df$bmi_category == 2,]
BMI_group_2_mean <- mean(df_group_2$k5_ti_adj)
BMI_group_2_error <- sd(df_group_2$k5_ti_adj) /
  sqrt(length(df_group_2))

df_group_3 <- runners_anova_df[runners_anova_df$bmi_category == 3,]
BMI_group_3_mean <- mean(df_group_3$k5_ti_adj)
BMI_group_3_error <- sd(df_group_3$k5_ti_adj) /
  sqrt(length(df_group_3))

groups = c(1, 2, 3)
mean_values = c(BMI_group_1_mean, BMI_group_2_mean, BMI_group_3_mean)
error_values = c(BMI_group_1_error, BMI_group_2_error, BMI_group_3_error)

# Create the bar plot with error bars
box_plot = ggboxplot(runners_anova_df, x="bmi_category", y="k5_ti_adj",
                     color="bmi_category", palette=c("#00AFBB", "#E7B800", "#FC4E07"),
                     order=c("1", "2", "3"),
                     ylab="5K Time in Seconds", xlab="BMI Range"
                     ) +
  scale_x_discrete(labels = c("0-22.3", "22.3-24.5", "24.5+")) +
  theme(panel.border = element_rect(colour="black", fill=NA, size=1)) +
  labs(color="BMI Category")

#Save plot to output
ggsave("RunnerBMIResults/boxplot_runners_5k_BMI.pdf", plot=box_plot)


pairwise_t_test_bonferroni <- pairwise.t.test(runners_anova_df$k5_ti_adj, 
                                   runners_anova_df$bmi_category,
                                   p.adjust.method = "bonferroni")

pairwise_t_test_BH <- pairwise.t.test(runners_anova_df$k5_ti_adj, 
                                   runners_anova_df$bmi_category,
                                   p.adjust.method = "BH")

#Print the results of the tests to a file.
sink("RunnerBMIResults/Pairwise-t-test-results.txt")
print(pairwise_t_test_bonferroni)
print(pairwise_t_test_BH)
sink()

########################  ANOVA and pairwise test discussion  ########################

############  ANOVA ############ 

#F = 27.607
#p = 2.795e-11
# A larger F-test indicates strong evidence to reject the null hypothesis (which states
# that there are no significant difference or no significant relationship in the context
# of the analysis). I.e. a higher F-test represents more explained variance, which
# means that the samples were not just drawn from a random sample of identical
# distributions, rather, the samples were drawn randomly from different distributions.
# The value of 27.607 indicates strong evidence that we can reject our null hypothesis
# and that the variances in a runner's 5K time are dependent on their BMI. This is further
# supported by an extremely low p-value. Typical thresholds of p-values are ~0.05,
# where we can reject our null hypothesis. We have a p-value that is roughly 9 orders 
# of magnitude lower, showing that this large F-test value is valid.

############  Pairwise T-test - Bonferroni  ############

#BMI Group 1 vs. BMI Group 2: p=1.00
#BMI Group 1 vs. BMI Group 3: p=6.2e-12
#BMI Group 2 vs. BMI Group 3: p=5.7e-11

# The results of this pairwise-t-test show that BMI group 1 and BMI group 2 have
# extremely similar means and error. A p value of 1.00 fails to reject the null
# hypothesis, and therefore BMI Group 1 and BMI Group 2 contain no significant 
# difference in the context of their mean and variance. On the other hand, BMI
# Group 1 and BMI Group 2 have a p value of 6.2e-12 and BMI Group 2 and BMI Group 3
# contain a p value of 5.7e-11. This indicates that we can reject the null hypothesis
# for these comparisons. Essentially runners with a BMI lower than 24.5 can all be
# considered a part of the same distribution, whereas once the BMI increases over 24.5
# the range of variance and the mean begin to significantly differ. The BMI groups
# were selected in such a manner that the entire dataset was subdivided into 3 groups
# where each group would have roughly an equal number of samples. This explains
# why Group 1 and Group 2 have similar distributions, the divide was arbitrary to
# allow even groupings, without general physical intuition of what a "bad" or "good"
# BMI would be.

############  Pairwise T-test - Benjamini-Hochberg (BH)  ############

#BMI Group 1 vs. BMI Group 2: p=0.72
#BMI Group 1 vs. BMI Group 3: p=6.2e-12
#BMI Group 2 vs. BMI Group 3: p=2.9e-11

# The overarching analysis remains the same as the Bonferroni method. Yet, differences
# in Group 1 vs Group 2 and Group 2 vs Group 3 are observed. The Bonferroni method
# controls the familywise error rate, which is the probability of making at least one
# false positive among a set of multiple hypothesis tests whereas the BH method
# controls the false discovery rate, which gives the expected proportion of false
# positives among all significant results. This allows for a higher number of false
# positive results than compared to the Bonferroni method. Hence, we see a decrease in
# p-values (aside from Group 1 vs Group 2 due to their extreme similarity). In short
# The Bonferroni method is more sensitive to false positives then BH.

########################  End discussion  ########################

#Perfrom Kruskal Wallis Test
#1) Rank all dependent variables, regardless of group
#2) Assemble these variables back into their groups, where their ranks from the
#   overall rankings remain
#)  Critical Value of chi^2 given our DOF and probability of 0.05: 5.99

kruskal_wallis_test <- kruskal.test(runners_anova_df$k5_ti_adj ~ runners_anova_df$bmi_category)
#H = 35 > 5.99, therefore we reject the null hypothesis, and the medians of the groups
#are not all the same, and some groups do not come from the same distribution.


#Conduct Pearson and Spearman Correlations

#Pearson correlation assumes the data have a monotonic linear relationship and
#that the data is normally distributed.
correlation_pearson <- cor(runners_df_clean$bmi,
                          runners_df_clean$k5_ti_adj,
                          method='pearson')

#Spearman correlation is a non-parametric test that does not assume a specific type
#of distribution for the data and can capture non-linear data.
correlation_spearman <- cor(runners_df_clean$bmi,
                           runners_df_clean$k5_ti_adj,
                           method='spearman')

#The Pearson and Spearman correlation give values of 0.46 and 0.37 respectively, 
#indicating some correlation between BMI and 5K times. With the Pearson correlation
#being higher it lends one to believe the data might have some sense of a normal
#distribution. Both are positive indicating that an increase in BMI is associated
#with slower 5K times. Yet, our Kruskal-Wallis test indicates that some groups are
#not associated with the same distribution which come from higher BMIs, Which create
#outliers in our distribution as we can see in the histogram "Runner Times" where
#the distribution is positively skewed skewed 

runner_times = unique(runners_df_clean$k5_ti_adj)
#Null Hypothesis that the data comes from a normal distribution
#Critical Value to compare D to: alpha = 0.05 N >> 35: 1.36/sqrt(N) = 0.044
ks_test = ks.test(runner_times, 'pnorm', mean=mean(runner_times), sd=sd(runner_times))
sink('RunnerBMIResults/correlation_test_results.txt')
print(kruskal_wallis_test)
print(paste('Pearson Correlation: ', correlation_pearson))
print(paste('Spearman Correlation: ', correlation_spearman))
print(ks_test)
sink()
#THE KS test returned a statistic of 0.081 and a p-value of ~8.1e-6. The statistic
#value of 0.081 > 0.044 and with our extremely low p-value we can say with confidence
#that our data is normally distributed, which lends insight into as why our
#Pearson Correlation was slightly stronger than our Spearman Correlation.

#All tests seem to agree with one another that BMI is in fact correlated with 
#runner's 5K times. The successful KS test with a 0.46 Pearson correlation indicate
#that increasing BMI increases 5K times and the data is normally distributed.

runners_correlation = ggplot(runners_df_clean, aes(x = runners_df_clean$bmi,
                             y = runners_df_clean$k5_ti_adj))+
                        geom_point(shape=19, color='blue') +
                        labs(
                          title='BMI vs 5K Time',
                          x = "BMI",
                          y = "5K Time (seconds)"
                        ) + 
                        geom_text(
                          x = min(runners_df_clean$bmi),
                          y = max(runners_df_clean$k5_ti_adj),
                          label = paste("Pearson Correlation =", round(correlation_pearson,2)),
                          hjust = 0,
                          vjust = 1
                        ) +
                        geom_text(
                          x = min(runners_df_clean$bmi),
                          y = max(runners_df_clean$k5_ti_adj),
                          label = paste("Spearman Correlation =", round(correlation_spearman,2)),
                          hjust = 0,
                          vjust = 2
                        )

runners_histo = ggplot(data = NULL, aes(x = runner_times))+
                  geom_histogram(binwidth=50, fill="blue", color='black')+
                  labs(title = "Runner Times", x = 'K5 Time (Adjusted)', y='Freq')


ggsave("RunnerBMIResults/Runners_5K_BMI_Correlations.pdf", plot=runners_correlation)
ggsave('RunnerBMIResults/Runners_5K_Distribution.pdf', plot=runners_histo)

#Run a simple Linear Regression on BMI vs 5K Times
linear_model <- lm(runners_df_clean$bmi ~ runners_df_clean$k5_ti_adj, data=runners_df_clean)
linear_model_summary <- summary(linear_model)

#Save the model summary
sink('RunnerBMIResults/linear_model_BMI_5K.txt')
print(linear_model_summary)
sink()

runners_regression = ggplot(runners_df_clean, aes(x = runners_df_clean$bmi,
                                                  y = runners_df_clean$k5_ti_adj))+
  geom_point(shape=19, color='blue') +
  labs(
    title='BMI vs 5K Time',
    x = "BMI",
    y = "5K Time (seconds)"
  ) + 
  geom_text(
    x = min(runners_df_clean$bmi),
    y = max(runners_df_clean$k5_ti_adj),
    label = paste("R^2 =", round(linear_model_summary$r.squared, 2)),
    hjust = 0,
    vjust = 1
  )

ggsave('RunnerBMIResults/Runners_5k_BMI_Regression.pdf', plot=runners_regression)

#The R-squared is much lower than the correlation results. From this relationship
#we can say that roughly 21% of the explaination of increased 5K times come from
#BMI. We would want to use regression when we want to make a prediction or forecast
#some future event, whereas if we are solely interested in the strength and direction
#of correlation we should use other correlation based tests.






