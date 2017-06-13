library(forecast)
library(tseries)
library(fitdistrplus)
require(graphics)

mydata <- read.csv('dataset_time_series.csv', header = TRUE, sep=";", dec = ",")
colnames(mydata)
mydata[,c("N","X","X.1","X.2","X.3","X.4","X.5")] <- list(NULL)

dat2_time_series <- ts(data=subset(mydata, select=c("Dat2")))
plot(dat2_time_series, main="Time Series for Dat2 Values",
     xlab="Time",ylab="Amount", col="blue") 

summary(dat2_time_series)
sd(dat2_time_series)

dat2_as_vector = as.vector(dat2_time_series)
plot(x = seq(1, length(dat2_as_vector)), y = dat2_as_vector)

cor(x = seq(1, length(dat2_as_vector)), y = dat2_as_vector) # == 1
