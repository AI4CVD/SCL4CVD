library(readxl)
library(ggplot2)
library(gcookbook)
library(svglite)
library(reshape2)

data <- read_excel("Paper2_rdrop_alpha.xlsx", sheet = "F-score")
data <- melt(data,id='α')
colnames(data) <- c("α","metrics","value")

pg_line <-
  ggplot(data,
         aes(
           x = α,
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
  theme(legend.title = element_blank())+ scale_x_continuous(breaks = seq(1,9,2))
pic