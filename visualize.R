library(MASS)
library(base)
library(ggplot2)
library(tidyr)
library(dplyr)
library(scales)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(readr)

my_data <- read.csv('result_HLL.csv', stringsAsFactors = FALSE)

wkRev <- my_data %>% 
  pivot_longer(-Week, names_to = 'pricing.scheme', values_to = 'avg.weekly.revenue')

wk_rev_plot <- ggplot(data = wkRev,
                      aes(x = Week, y = avg.weekly.revenue, 
                          group = pricing.scheme, 
                          shape = pricing.scheme))
rev_plot <- wk_rev_plot + geom_line(aes(linetype = pricing.scheme,
                                        color = pricing.scheme))+
  geom_point(aes(color = pricing.scheme)) +
  scale_x_continuous(name = 'Week') +
  scale_y_continuous(name = 'Avg. Revenue',
                     labels = dollar) +
  theme(legend.position = c(0.7, 0.5),
        legend.direction = 'horizontal',
        legend.title = element_blank(),
        plot.title = element_text(size = 12, face = 'bold.italic', hjust = 0.5),
        axis.title.x = element_text(color = 'black', size = 11, face = 'bold'),
        axis.title.y = element_text(color = 'black', size = 11, face = 'bold'),
        panel.border = element_rect(colour = "black", fill=NA))
rev_plot
