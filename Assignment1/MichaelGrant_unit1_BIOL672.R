#Michael Grant
#Operating System: Windows 10/11
#Packages/Libraries: ggplot2, dplyr, ggpubr
#Data Files: "RunnerData.csv"

library(ggplot2)
library(ggpubr)
library(dplyr)

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

invisible(sink("desc.txt"))

cat(output, sep="\n")

sink()

#Create a PDF of the generated plot 
ggsave("histo.pdf", plot=histogram_poisson_distribution)

#Load in the csv for the One-way ANOVA - I will be testing three groups of runners
#data. Each subgroup will be within a specific BMI range, and the null hypothesis
#is that by randomly selecting from each group, the average 5K time should remain
#the same.

#Reads in csv file and saves as a dataframe
runners_df <- read.csv("RunnerData.csv")

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
sink('ANOVA_RESULTS.txt')
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
ggsave("boxplot_runners_5k_BMI.pdf", plot=box_plot)


pairwise_t_test_bonferroni <- pairwise.t.test(runners_anova_df$k5_ti_adj, 
                                   runners_anova_df$bmi_category,
                                   p.adjust.method = "bonferroni")

pairwise_t_test_BH <- pairwise.t.test(runners_anova_df$k5_ti_adj, 
                                   runners_anova_df$bmi_category,
                                   p.adjust.method = "BH")

#Print the results of the tests to a file.
sink("Pairwise-t-test-results.txt")
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







