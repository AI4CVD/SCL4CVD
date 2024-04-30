library(readxl)
library(ggplot2)
library(gcookbook)
library(svglite)
library(reshape2)

data <- read_excel("Paper2_dropout.xlsx", sheet="F-score")
data <- melt(data,id='dropout')
colnames(data) <- c("dropout","metrics","value")

pg_line <-
  ggplot(data,
         aes(
           x = dropout,
           y = value,
           colour = metrics,
           group = metrics,
           shape = metrics
         )) +
  geom_line() + ylim(0, 1) + geom_point(size=3)+xlab(NULL)+ylab(NULL)

pic <- pg_line + theme_bw()+
  theme(legend.position = c(.045, .045),
        legend.justification = c(0, 0),
        legend.text = element_text(face="bold", size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15)) +
  theme(legend.background = element_rect(fill = "white", colour = "black")) +
  theme(legend.title = element_blank())+ scale_x_continuous(breaks = c(0.01,0.03,0.05,0.07,0.09))
pic