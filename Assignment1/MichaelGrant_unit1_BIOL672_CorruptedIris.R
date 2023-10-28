#Michael Grant
#Operating System: Windows 10/11
#Packages/Libraries: ggplot2, dplyr, ggpubr, dgof, gridExtra
#Data Files: "input_data/iris_original.txt", "input_data/iris_tab_bgnoise.txt",
#            "input_data/iris_tab_smnoise.txt", "input_data/iris_corrupted.csv" 

library(ggplot2)
library(ggpubr)
library(dplyr)
library(dgof)
library(gridExtra)

setwd("C:/Users/Grant/OneDrive/Homework/Fall 2023/BIOL672/Assignment1")

# Load in the original not corrupted iris dataset
original_iris_data = read.table('input_data/iris_original.txt', sep='\t', header=TRUE)

# Load the three corrupted datasets
iris_bignoise_data = read.table('input_data/iris_tab_bgnoise.txt', sep='\t', header=TRUE)
iris_smallnoise_data = read.table('input_data/iris_tab_smnoise.txt', sep='\t', header=TRUE)
iris_corrupted_data = read.table('input_data/iris_corrupted.csv', sep=',', header=TRUE)

# While these datasets represent the same type of data, there should be some
# discrepancies amongst the data. Each dataset has sepal length/width and 
# petal length width. I will conduct an ANOVA for each of these metrics within
# the sample sets.

# To do this I will create a master data frame of the 4 metrics and then the 
# categorical value will be which dataset it came from. Therefore, the dataframe
# will conatin the following columns:
# sepal_length, sepal_width, petal_length, petal_width, dataset
# The datasets will be labelled as:
# original_iris_data = 1
# iris_bignoise_data = 2
# iris_smallnoise_data = 3
# iris_corrupted_data = 4

original_iris_data$dataset <- 1
iris_bignoise_data$dataset <- 2
iris_smallnoise_data$dataset <- 3
iris_corrupted_data$dataset <- 4

master_iris_data <- rbind(original_iris_data, iris_bignoise_data, iris_smallnoise_data, iris_corrupted_data)


# We can run a MANOVA and observe broadly if the variance between samples able to distinguish
# between datasets.
manova_iris_data <- manova(cbind(sepal_length, sepal_width, petal_length, petal_width) ~ dataset, 
                           data = master_iris_data)

sink('CorruptedIrisData/MANOVA_Iris.txt')
print (summary(manova_iris_data))
sink()

# The MANOVA test returned an F-value of 2.8367 - which is closer to one, supporting our 
# null hypothesis that the variation among metric means is due more to chance than
# it is an explained variance - which makes sense as though noise/corruption is introduced
# into the datasets, the general predictions should remain the same, albeit with an
# increase/decrease in variance

# We can further run one-way ANOVA's too examine each individual metric's effect on
# discerning if they come from the same dataset
anova_sepal_length <- oneway.test(master_iris_data$sepal_length ~ master_iris_data$dataset)
anova_sepal_width <- oneway.test(master_iris_data$sepal_width ~ master_iris_data$dataset)
anova_petal_length <- oneway.test(master_iris_data$petal_length ~ master_iris_data$dataset)
anova_petal_width <- oneway.test(master_iris_data$petal_width ~ master_iris_data$dataset)

sink('CorruptedIrisData/ANOVA_Sepal_Length_Iris.txt')
print(anova_sepal_length)
sink()

#F-value: 11.779

sink('CorruptedIrisData/ANOVA_Sepal_Width_Iris.txt')
print(anova_sepal_width)
sink()

#F-value: 33.753

sink('CorruptedIrisData/ANOVA_Petal_Length_Iris.txt')
print(anova_petal_length)
sink()
#F-value: 3.1102

sink('CorruptedIrisData/ANOVA_Petal_Width_Iris.txt')
print(anova_petal_width)
sink()
#F-value: 16.334

# It appears that individually we can reject the null-hypothesis for each metric except
# petal length. Therefore, it would appear that the variations of mean for each individual 
# metric are due to more than chance and we can assume that Dr. G gave us falsified data

# Import the extended iris data sets to analyze the categorical and ordinal results from
# the purchase of iris flowers by customers
iris_purchased_data = read.table('input_data/iris_csv_purchase.csv', sep=',', header=TRUE)

iris_purchased_data$color[iris_purchased_data$color == "yellow"] <- 1
iris_purchased_data$color[iris_purchased_data$color == "orange"] <- 2
iris_purchased_data$color[iris_purchased_data$color == "red"] <- 3
iris_purchased_data$color[iris_purchased_data$color == "blue"] <- 4

iris_purchased_data$species[iris_purchased_data$species == "setosa"] <- 1
iris_purchased_data$species[iris_purchased_data$species == "virginica"] <- 2
iris_purchased_data$species[iris_purchased_data$species == "versicolor"] <- 3

iris_purchased_data$sold[iris_purchased_data$sold == "FALSE"] <- 0
iris_purchased_data$sold[iris_purchased_data$sold == "TRUE"] <- 1

iris_purchased_data$likelytobuy[iris_purchased_data$likelytobuy < 0] <- 0
iris_purchased_data$likelytobuy[iris_purchased_data$likelytobuy > 0] <- 1

chisq_likely_sold <- chisq.test(iris_purchased_data$likelytobuy, iris_purchased_data$sold, correct=FALSE)

chisq_results_color <- chisq.test(iris_purchased_data$color, iris_purchased_data$sold, correct=FALSE)
chisq_results_species <- chisq.test(iris_purchased_data$species, iris_purchased_data$sold, correct=FALSE)
chisq_results_attractiveness <- chisq.test(iris_purchased_data$attractiveness, iris_purchased_data$sold, correct=FALSE)
chisq_results_review <- chisq.test(iris_purchased_data$review, iris_purchased_data$sold, correct=FALSE)

sink('CorruptedIrisData/chisqTesting.txt')
print(chisq_likely_sold)
sink()

# First I wanted to observe if a flower was likely to be bought if it actually sold or not
# the chi squared result was 0.8336 with a p-value of 0.3612 which means that most variability
# is random and if we know a flower is likely to be bought then it will be.

# I then want to use the data available to predict whether or not a flower is likely 
# to be bought. I can use a regression model for the varying continuous values within 
# a logistic regression based upon varying categorical data points

#Regression for each color
yellow_regression <- iris_purchased_data[iris_purchased_data$color == 1,]

yellow_outcome = yellow_regression$likelytobuy

yellow_regression <- data.frame(
  SL = c(yellow_regression$sepal_length),
  SW = c(yellow_regression$sepal_width),
  PL = c(yellow_regression$petal_length),
  PW = c(yellow_regression$petal_width),
  LTB = c(yellow_regression$likelytobuy)
)

scaled_data <- as.data.frame(apply(yellow_regression[, sapply(yellow_regression, is.numeric)], 2, function(x) (x - min(x)) / (max(x) - min(x))))


model_yellow <- glm(LTB ~ SL + SW + PL + PW,
                    data = scaled_data, family = binomial('logit'))


scaled_data$prediction <- predict(model_yellow, newdata = scaled_data, type = "response")

plot1 <- ggplot(scaled_data, aes(x = SL, y = prediction)) +
                  geom_line()

plot2 <- ggplot(scaled_data, aes(x = SW, y = prediction)) +
  geom_line()

plot3 <- ggplot(scaled_data, aes(x = PL, y = prediction)) +
  geom_line()

plot4 <- ggplot(scaled_data, aes(x = PW, y = prediction)) +
  geom_line()

pdf('CorruptedIrisData/YellowLogisticRegression.pdf')
grid.arrange(plot1, plot2, plot3, plot4)
dev.off()


blue_regression <- iris_purchased_data[iris_purchased_data$color == 4,]

blue_outcome = blue_regression$likelytobuy

blue_regression <- data.frame(
  SL = c(blue_regression$sepal_length),
  SW = c(blue_regression$sepal_width),
  PL = c(blue_regression$petal_length),
  PW = c(blue_regression$petal_width),
  LTB = c(blue_regression$likelytobuy)
)

scaled_data <- as.data.frame(apply(blue_regression[, sapply(blue_regression, is.numeric)], 2, function(x) (x - min(x)) / (max(x) - min(x))))


model_blue <- glm(LTB ~ SL + SW + PL + PW,
                    data = scaled_data, family = binomial('logit'))


scaled_data$prediction <- predict(model_blue, newdata = scaled_data, type = "response")

plot1 <- ggplot(scaled_data, aes(x = SL, y = prediction)) +
  geom_line()

plot2 <- ggplot(scaled_data, aes(x = SW, y = prediction)) +
  geom_line()

plot3 <- ggplot(scaled_data, aes(x = PL, y = prediction)) +
  geom_line()

plot4 <- ggplot(scaled_data, aes(x = PW, y = prediction)) +
  geom_line()

pdf('CorruptedIrisData/BlueLogisticRegression.pdf')
grid.arrange(plot1, plot2, plot3, plot4)
dev.off()

# In the previous analysis I used petal length, petal width, sepal length and sepal width
# in a logistic regression to analyze how these features are able to predict the 
# probability of a flower to be likely to be bought. While yellow flowers showed trends in
# probability such as smaller sepal width, petal length and petal width are more likely to sell
# while larger sepal width increase the probability of selling, the overall chance of
# a yellow flower selling is relatively low. On the other hand, Blue flowers show no real
# trend in leaf attributes, but overall have higher probabilitis of selling. This indicates
# thata people seem to care more about flower color than the leafs associated with the flower

