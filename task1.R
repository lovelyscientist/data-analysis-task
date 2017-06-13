library(forecast)
library(tseries)
require(graphics)

mydata <- read.csv('dataset_time_series.csv', header = TRUE, sep=";", dec = ",")
colnames(mydata)
mydata[,c("N","X","X.1","X.2","X.3","X.4","X.5")] <- list(NULL)

dat1_time_series <- ts(data=subset(mydata, select=c("Dat1")))
plot(dat1_time_series,  main="Time Series for Dat1 Values",
     xlab="Time",ylab="Amount") 

dat2_time_series <- ts(data=subset(mydata, select=c("Dat2")))
plot(dat2_time_series, main="Time Series for Dat2 Values",
     xlab="Time",ylab="Amount") 

mf <- meanf(dat1_time_series,h=100)
plot(mf) 
adf1 <- adf.test(dat1_time_series)
adf1

adf2 <- adf.test(dat2_time_series)
adf2

acf(dat1_time_series, type="correlation")
acf(dat2_time_series, type="correlation")


ndiffs(dat1_time_series[,1])
diff_data <- diff(dat1_time_series[,1])

forecast(auto.arima(dat1_time_series, stationary=TRUE, seasonal=FALSE))
plot(forecast(auto.arima(dat1_time_series, stationary=TRUE, seasonal=FALSE)))

