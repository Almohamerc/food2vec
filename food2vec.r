library(ggplot2)

data <- read.csv('foods.csv', header=FALSE)
my.plot <- ggplot(data, aes(x=V2, y=V3, label=V1)) + geom_text(size=3)
