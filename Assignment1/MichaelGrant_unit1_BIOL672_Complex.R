#Michael Grant
#Operating System: Windows 10/11
#Packages/Libraries: ggplot2, dplyr, ggpubr, dgof, psych, mixtools, MASS, grid
#Data Files: "input_data/BostonHousingPricesResults.csv"

library(ggplot2)
library(ggpubr)
library(dplyr)
library(dgof)
library(psych)
library(mixtools)
library(MASS)
library(grid)

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
#NOX - nitric oxide concentration (parts per 10 million)
#RM - average rooms per dwelling
#AGE - proportion of owner-occupied units built prior to 1940
#DIS - distance to five Boston employment centers
#TAX - full-value property-tax rate per $10,000
#PTRATIO - pupil-teacher ratio
#B - 1000(Bk-0.63)^2 where Bk is the proportion of blacks by town
#LSTAT - lower status of the population
#MEDV - median value of owner occupied homes in $1000s
columns = c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
            'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV')

colnames(boston_housing_price_data) <- columns

PCAs = c('CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 
         'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV')

boston_housing_price_data <- boston_housing_price_data[PCAs]

scaled_BH_price_data <- scale(boston_housing_price_data)

pca_BH_price <- prcomp(scaled_BH_price_data)

pca_loadings <- pca_BH_price$rotation

scree_data <- data.frame(PC = 1:ncol(scaled_BH_price_data), VarianceExplained = pca_BH_price$sdev^2)

scree_data$VarianceExplained <- scree_data$VarianceExplained / sum(scree_data$VarianceExplained)

scree_plot <- ggplot(scree_data, aes(x = PC, y = VarianceExplained)) +
  geom_line() +
  labs(x = "Principal Component (PC)", y = "Variance Explained") +
  scale_x_continuous(breaks = 1:ncol(scaled_BH_price_data))

pdf('FactorAnalysis/ScreePlot.pdf')
grid.arrange(scree_plot)
dev.off()

sink('FactorAnalysis/PCAAnalysis.txt')
print(pca_loadings)
print(summary(pca_BH_price))
sink()

# The factor analysis shows that to only 3 PCs are needed to account for ~72% of
# the variance within the data. These factors are the crime rate, proportion of
# residential land zoned for lots over 25,000sqft and #INDUS - proportion of
# non-retail business acres per town. Therefore the reduction of data was successful

# For the PCs that explain more variance it appears that each variable plays an important
# role. Each component has a magnitude >0.2. INDUS, NOX, AGE, TAX and LSTAT seem to have
# the greatest loadings. PC2 is interesting because it has much stronger loadings for RM
# and MEDV, meaning that this PC seems dependent on the size and cost of the house.
# PC3 seems mainly driven by CRIM and ZN. The general trend is that the first few PCs
# seem to be effected by each metrics loading, whereas we decrease in PC individual metrics
# are more pronounced.

# PC4 is strongly driven by PTRATIO, PC5 is strongly driven by the proportion of population
# that is black, PC7 is strongly influenced by AGE, PC8 is strongly driven by MEDV,
# PC9 -> LSTS, PC10 -> INDUS, PC11 DIS PCA12 is NOX and DIs



#FACTOR ANALYSIS OF SAME DATA
factor_analysis_BH_prices <- fa(boston_housing_price_data, nfactors=4, rotate = "varimax", fm = 'ml')

sink("FactorAnalysis/FactorAnalysis.txt")
print(factor_analysis_BH_prices)
sink()


# In this factor analysis it appears that INDUS and NOX and AGE seem to cluster together
# While some loadings between factors are similar, each factor has variables that
# differ drastically from others - making each latent varible relatively unique.
# ML1, ML2 and ML3 are all significant as their eigen values are > 1 while ML4 is not
# significant. It appears that INDUS and NOX seem to cluster together and aside from that
# the clustering is pretty random. If two traits are far apart on the axis of significant
# factor in a Factor Analysis then it indicates that these variables have a weak
# relationship whereas if they are close then it suggests the traits are related.
# Yes, the first three latent variables are able to describe roughly 90% of the variance
# within the data.

#CLUSTERING OF PCS 1 and 2
PCA_DATA <- data.frame(pca_BH_price$x[, c('PC1', 'PC2')])

pdf('FactorAnalysis/InitialScatter.pdf')
scatter <- plot(PCA_DATA$PC1, PCA_DATA$PC2, type = "p", pch = 15, col = "black", xlab = "PC1", ylab = "PC2")
dev.off()

#From this scatter plot I could see roughly 3 clusters

kmeans_result <- kmeans(PCA_DATA, centers = 3)

cluster_assignments <- kmeans_result$cluster

cluster_centers <- kmeans_result$centers

pdf('FactorAnalysis/ClusterScatter.pdf')
plot(PCA_DATA, col = cluster_assignments, pch=15)
points(cluster_centers, col=1:3, pch=3, cex=2)
dev.off()

#The resulting plot shows three different clusters but not how I thought it was going to be
# The green cluster is one I predicted,The top of the black cluster, between
# [-4,0] for PC1 and [2,4] for PC2 I thought would be a cluster and the remaining would
# be the third cluster

#Now using Crime we will make a normal distribution, lognormal and exponential
independent_variable <- boston_housing_price_data$CRIM
n <- length(independent_variable)

normal_distribution <- fitdistr(independent_variable, densfun="normal")
lognormal_distribution <- fitdistr(independent_variable, densfun="log-normal")
exp_distribution <- fitdistr(independent_variable, densfun="exponential")
GMM_distribution <- normalmixEM(independent_variable, k=3)

BIC_normal <- -2 * normal_distribution$loglik+1*log(n)
BIC_log <- -2 * lognormal_distribution$loglik+1*log(n)
BIC_exp <- -2 * exp_distribution$loglik+1*log(n)
BIC_GMM <- -2 * GMM_distribution$loglik+3*log(n)

sink('FactorAnalysis/BIC_Comparisons')
print('Normal Distribution')
print(BIC_normal)
print("Log-normal Distribution")
print(BIC_log)
print("Exponential Distribution")
print(BIC_exp)
print("GMM")
print(BIC_GMM)
sink()

plotting_frame <- data.frame(independent_variable)

normal_plot <- ggplot(plotting_frame, aes(x=independent_variable)) +
  geom_histogram(binwidth = 100, aes(y=..density..)) +
  geom_density() +
  stat_function(fun=dnorm, color="red", args=list(mean = normal_distribution$estimate[1], sd = normal_distribution$estimate[2]))

log_plot <- ggplot(plotting_frame, aes(x=independent_variable)) +
  geom_histogram(binwidth = 100, aes(y=..density..)) +
  geom_density() +
  stat_function(fun=dlnorm, color="red", args=list(meanlog = lognormal_distribution$estimate[1], sdlog = lognormal_distribution$estimate[2]))

exp_plot <- ggplot(plotting_frame, aes(x=independent_variable)) +
  geom_histogram(binwidth = 100,aes(y=..density..)) +
  geom_density() +
  stat_function(fun=dexp, color="red", args=list(rate = exp_distribution$estimate[1]))

GMM_plot <- ggplot(plotting_frame, aes(x=independent_variable)) +
  geom_histogram(binwidth = 100,aes(y=3*(..density..))) +
  geom_density() +
  stat_function(fun=dnorm, color='red', args=list(mean = GMM_distribution$mu[1], sd = GMM_distribution$sigma[1]))+
  stat_function(fun=dnorm, color='red', args=list(mean = GMM_distribution$mu[2], sd = GMM_distribution$sigma[2]))+
  stat_function(fun=dnorm, color='red', args=list(mean = GMM_distribution$mu[3], sd = GMM_distribution$sigma[3]))

pdf('FactorAnalysis/GMMPlots.pdf')
pushViewport(viewport(layout = grid.layout(2,2)))
print(normal_plot, vp = viewport(layout.pos.row=1, layout.pos.col = 1))
print(log_plot, vp = viewport(layout.pos.row=1, layout.pos.col = 2))
print(exp_plot, vp = viewport(layout.pos.row=2, layout.pos.col = 1))
print(GMM_plot, vp = viewport(layout.pos.row=2, layout.pos.col = 2))
dev.off()

# From the model fitting we see that the log-normal distribution fits our data the best
# The normal distribution fits it the worst. The GMM model did not perform the best,
# which means that using 3 density functions is not sufficient in the development
# of latency. If we increase the number of density functions we can achieve latency


GMM_distribution_k5 <- normalmixEM(independent_variable, k=5)
BIC_GMM_k5 <- -2 * GMM_distribution_k5$loglik+5*log(n)
#Gives a BIC score of 1385, which would be the best score

# By increasing the number of density functions used we can generate a better fit
# Arbitrarily increasing the density functions can lead to overfitting which would
# render the model useless.

