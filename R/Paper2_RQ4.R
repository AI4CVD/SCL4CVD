library(readxl)
library(ggplot2)
library(gcookbook)
library(svglite)
library(reshape2)
library(viridis)

data <- read_excel("Paper2_RQ4.xlsx", sheet="F1")
data <- melt(data,id='model')
colnames(data) <- c("model","portion","value")

pg_line <-
  ggplot(data,
         aes(
           x = portion,
           y = value,
           colour = model,
           group = model,
           shape = model
         ))+
  geom_line(linewidth=0.8) + ylim(0.0, 0.7) + geom_point(size=3)+xlab(NULL)+ylab(NULL)

pic <- pg_line + theme_bw()+
  theme(legend.position = c(.5, .005),
        legend.justification = c(0, 0),
        legend.text = element_text(face="bold", size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15)) +
  theme(legend.background = element_rect(fill = "white", colour = "black")) +
  theme(legend.title = element_blank())
pic