#Michael Grant
#Operating System: Windows 10/11
#Packages/Libraries: ggplot2, dplyr, ggpubr, dgof
#Data Files: "input_data/BostonHousingPricesResults.csv"

library(ggplot2)
library(ggpubr)
library(dplyr)
library(dgof)

setwd("C:/Users/Grant/OneDrive/Homework/Fall 2023/BIOL672/Assignment1")

#For this portion of the unit assignment I will be using the Boston Housing Price
#dataset.

#Read in csv file containing housing data - the csv downloaded from Kaggle does not
#contain any column headers.
boston_housing_price_data <- read.table("input_data/housing.csv", header = FALSE)

#Change column names to represent the actual data
#CRIM - per capita crime rate by town
#ZN - proportion of residential land zoned for lots over 25,000sqft
#INDUS - proportion of non-retail business acres per town
#CHAS - 1 if tract bounds river, 0 otherwise
#NOX - nitric oxide concentration (parts per 10 million)
#RM - average rooms per dwelling
#AGE - proportion of owner-occupied units built prior to 1940
#DIS - distance to five Boston employment centers
#RAD - index of accessibility to radial highways
#TAX - full-value property-tax rate per $10,000
#PTRATIO - pupil-teacher ratio
#B - 1000(Bk-0.63)^2 where Bk is the proportion of blacks by town
#LSTAT - lower status of the population
#MEDV - median value of owner occupied homes in $1000s
colnames(boston_housing_price_data) <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
                                        'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV')

#Reads the first few lines of the dataset
head(boston_housing_price_data)

#I want to see if there is any significant difference between:
#Crime, Nitric Oxide Concentration, pupil-teacher ratio, tax rate and average rooms
#between different median income values

#Conduct the multivariate analysis of variance test
manova_boston <- manova(cbind(CRIM, NOX, AGE, TAX, PTRATIO) ~ MEDV,
                        data = boston_housing_price_data)

sink('BostonHousingPricesResults/MANOVA_Boston_Housing_Prices.txt')
print (summary(manova_boston))
sink()

# From these results we see that we have an extremely high F-statistic (63.65) meaning 
# that the means between the dependent variables are significantly different and they
# are not drawn from the same distributions. Therefore we can reject the null hypothesis
# that there is no significant difference/relationship in the context of the analysis.
# Crime, pollution, pupil-teacher ratio, tax and average rooms per dwelling all have
# an impact on the median value of housing in the Boston area. This is further verified
# by a near 0 p-value.

# Conduct a multiple regression that tells how the median home price is predicted 
# by the dependent variables used in the MANOVA.

multiple_regression_boston_housing <- lm(boston_housing_price_data$MEDV ~ 
                                           boston_housing_price_data$CRIM + 
                                           boston_housing_price_data$NOX + 
                                           boston_housing_price_data$AGE + 
                                           boston_housing_price_data$TAX + 
                                           boston_housing_price_data$PTRATIO)

sink('BostonHousingPricesResults/MultipleRegressionModelBostonHousing.txt')
print (summary(multiple_regression_boston_housing))
sink()

multiple_plot_matrix <- cbind(boston_housing_price_data$CRIM,
                              boston_housing_price_data$NOX,
                              boston_housing_price_data$AGE,
                              boston_housing_price_data$TAX,
                              boston_housing_price_data$PTRATIO)

# From the multiple linear regression we see an r^2 of 0.3828, meaning that roughly
# 62% of the variance in housing prices does not come from these 5 metrics. Furthermore,
# if we evaluate p-values we see that AGE and TAX does not appear to significantly
# impact the median value of houses at all, as there p-values are both > 0.75, which indicates
# we could just randomly gueses and have a better chance at predicting median house value.
# The other three metrics, CRIM, NOX and PTRATIO each have pvalues less than 0.05,
# indicating they do account for some variance within the data. Of these three metrics
# that can actually predict median housing price, I would estimate that CRIM would be the best predictor
# as it has an acceptable p-value and its standard error is the smallest. While
# NOX and PTRATIO have highest coefficents, they have standard error that is equal to
# or greater than these coefficents and therefore would tend to flucuate

# Strictly out of curiosity I will run a MLR on all variables within the data frame
# while ignoring AGE AND TAX, as we already know they are poor predictors.

multiple_regression_boston_housing_all <- lm(boston_housing_price_data$MEDV ~ 
                                           boston_housing_price_data$CRIM + 
                                           boston_housing_price_data$NOX + 
                                           boston_housing_price_data$ZN + 
                                           boston_housing_price_data$RM + 
                                           boston_housing_price_data$DIS +
                                           boston_housing_price_data$B +
                                           boston_housing_price_data$PTRATIO +
                                           boston_housing_price_data$LSTAT)

sink('BostonHousingPricesResults/AllVariablesMultipleRegressionModelBostonHousing.txt')
print (summary(multiple_regression_boston_housing_all))
sink()

# Interestingly when we take into account more metrics the r^2 increases to roughly
# 0.72, meaning the initially ignored metrics were able to account for a lot more 
# of the variance within the dataset. All pvalues are within the allowed threshold.
# I would say in this scenario the best predictor shifts to RM - which intuitively
# makes sense as the more rooms a house has, the larger is most likely is, which
# would drive the housing price up.

metric_labels <- c('CRIM', 'ZN', 'NOX', 'RM', 'DIS', 'PTRATIO', 'B', 'LSTAT')

multiple_plot_matrix <- cbind(boston_housing_price_data$CRIM,
                              boston_housing_price_data$NOX,
                              boston_housing_price_data$ZN,
                              boston_housing_price_data$RM,
                              boston_housing_price_data$DIS,
                              boston_housing_price_data$B,
                              boston_housing_price_data$PTRATIO,
                              boston_housing_price_data$LSTAT)

pdf('BostonHousingPricesResults/MLR_Results_DependentVariable_Comparison.pdf')
multiple_plot_boston_housing <- pairs(multiple_plot_matrix,
                                      col=as.factor(boston_housing_price_data$MEDV),
                                      labels=metric_labels)
dev.off()

# For the compositve variable analysis we will observe home prices as a function
# defined as MEDV = LSTAT*RM+ZN*DIS. I chose these variables because from the pairs plot
# we can see that there is a strong linear relation between LSTAT and RM and between
# ZN and DIS

ancova_boston <- aov(formula = boston_housing_price_data$MEDV ~
                       boston_housing_price_data$LSTAT*boston_housing_price_data$RM+
                       boston_housing_price_data$ZN*boston_housing_price_data$DIS)

sink('BostonHousingPricesResults/ANCOVAResultsBostonHousing.txt')
print (summary(ancova_boston))
sink()

# From this analysis we see that we may be better off just using each individual
# dependent variable rather than the composite function. Realistically from this
# we might be best off using any model that incorporates LSTAT, as the F-value is
# over 1000 with a very small pscore (near 0). While LSTAT*RM performs exceptionally well,
# ZN*DIS has an F-value of 3 coupled with a pvalue greater than 0.05, indicating that
# some of the variance comes is random.




