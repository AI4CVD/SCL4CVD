library(readxl)
library(ggplot2)
library(gcookbook)
library(svglite)
library(reshape2)

data <- read_excel("Paper2_RQ3.xlsx", sheet="fan")
data <- melt(data,id='model')
colnames(data) <- c("model","metrics","value")

pg_line <-
  ggplot(data,
         aes(
           x = model,
           y = value,
           colour = metrics,
           group = metrics,
           shape = metrics
         )) +
  geom_line() + ylim(0, 1) + geom_point()+xlab(NULL)+ylab(NULL)

pic <- pg_line + theme_bw()+
  theme(legend.position = c(.075, .045),
        legend.justification = c(0, 0)) +
  theme(legend.background = element_rect(fill = "white", colour = "black")) +
  theme(legend.title = element_blank())
pic