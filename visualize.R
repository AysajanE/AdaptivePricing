library(MASS)
library(base)
library(ggplot2)
library(tidyr)
library(dplyr)
library(scales)
library(reshape2)
library(gridExtra)
library(readr)
library(ggpubr)
#install.packages('ggpubr')

# Import simulated revenue data for all 8 different scenarios
data_HHH <- read.csv('result_HHH.csv', stringsAsFactors = FALSE)
# Add a new column, specifying the scenario in consideration
df_HHH <- cbind(scenario = 'HHH', data_HHH)

data_HHL <- read.csv('result_HHL.csv', stringsAsFactors = FALSE)
df_HHL <- cbind(scenario = 'HHL', data_HHL)

data_HLH <- read.csv('result_HLH.csv', stringsAsFactors = FALSE)
df_HLH <- cbind(scenario = 'HLH', data_HLH)

data_HLL <- read.csv('result_HLL.csv', stringsAsFactors = FALSE)
df_HLL <- cbind(scenario = 'HLL', data_HLL)

data_LHH <- read.csv('result_LHH.csv', stringsAsFactors = FALSE)
df_LHH <- cbind(scenario = 'LHH', data_LHH)

data_LHL <- read.csv('result_LHL.csv', stringsAsFactors = FALSE)
df_LHL <- cbind(scenario = 'LHL', data_LHL)

data_LLH <- read.csv('result_LLH.csv', stringsAsFactors = FALSE)
df_LLH <- cbind(scenario = 'LLH', data_LLH)

data_LLL <- read.csv('result_LLL.csv', stringsAsFactors = FALSE)
df_LLL <- cbind(scenario = 'LLL', data_LLL)

totalweeks = max(data_HHH['Week'])

# Tide the data so that it will be convient for plotting purpose
wkRev_HHH <- df_HHH[, -c(3:5)] %>% 
  pivot_longer(-c(scenario, Week), names_to = 'pricing.comp', values_to = 'avg.wkly.rev.gap')
wkRev_HHL <- df_HHL[, -c(3:5)] %>% 
  pivot_longer(-c(scenario, Week), names_to = 'pricing.comp', values_to = 'avg.wkly.rev.gap')
wkRev_HLH <- df_HLH[, -c(3:5)] %>% 
  pivot_longer(-c(scenario, Week), names_to = 'pricing.comp', values_to = 'avg.wkly.rev.gap')
wkRev_HLL <- df_HLL[, -c(3:5)] %>% 
  pivot_longer(-c(scenario, Week), names_to = 'pricing.comp', values_to = 'avg.wkly.rev.gap')

wkRev_LHH <- df_LHH[, -c(3:5)] %>% 
  pivot_longer(-c(scenario, Week), names_to = 'pricing.comp', values_to = 'avg.wkly.rev.gap')
wkRev_LHL <- df_LHL[, -c(3:5)] %>% 
  pivot_longer(-c(scenario, Week), names_to = 'pricing.comp', values_to = 'avg.wkly.rev.gap')
wkRev_LLH <- df_LLH[, -c(3:5)] %>% 
  pivot_longer(-c(scenario, Week), names_to = 'pricing.comp', values_to = 'avg.wkly.rev.gap')
wkRev_LLL <- df_LLL[, -c(3:5)] %>% 
  pivot_longer(-c(scenario, Week), names_to = 'pricing.comp', values_to = 'avg.wkly.rev.gap')

# Consider High (Low) demand intensity scenarios separately
wkRev_HD <- rbind(wkRev_HHH, wkRev_HHL, wkRev_HLH, wkRev_HLL)
wkRev_LD <- rbind(wkRev_LHH, wkRev_LHL, wkRev_LLH, wkRev_LLL)
# Change labels for different scenarios, instead of original simplified values in the data
levels(wkRev_HD$scenario) <- c("High Slopes, High Revenue Gap",
                               "High Slopes, Low Revenue Gap",
                               "Low Slopes, High Revenue Gap",
                               "Low Slopes, Low Revenue Gap")
levels(wkRev_LD$scenario) <- c("High Slopes, High Revenue Gap",
                               "High Slopes, Low Revenue Gap",
                               "Low Slopes, High Revenue Gap",
                               "Low Slopes, Low Revenue Gap")

# Use ggplot facet_wrap function to plot multiple graphs in one place
# High Demand Intensity
plt_HD <- ggplot(data = wkRev_HD, 
              aes(x = Week, y = avg.wkly.rev.gap, group = pricing.comp)) +
  geom_line(aes(linetype = pricing.comp, color = pricing.comp)) +
  geom_point(aes(color = pricing.comp, shape = pricing.comp), size=2) +
  scale_x_continuous(name = 'Week', breaks = seq(0, totalweeks, by = 5)) +
  scale_y_continuous(name = 'Weekly Revenue Gap (%)', 
                     breaks = extended_breaks(n = 10), labels = percent) +
  theme(plot.title = element_text(size = 12, face = 'bold.italic', hjust = 0.5),
        axis.title.x = element_text(color = 'black', size = 11, face = 'bold'),
        axis.title.y = element_text(color = 'black', size = 11, face = 'bold'),
        panel.border = element_rect(colour = "black", fill = NA)) +
  facet_wrap( ~ scenario, ncol=2)

# Low Demand Intensity
plt_LD <- ggplot(data = wkRev_LD, 
                 aes(x = Week, y = avg.wkly.rev.gap, group = pricing.comp)) +
  geom_line(aes(linetype = pricing.comp, color = pricing.comp)) +
  geom_point(aes(color = pricing.comp, shape = pricing.comp), size=2) +
  scale_x_continuous(name = 'Week', breaks = seq(0, totalweeks, by = 5)) +
  scale_y_continuous(name = 'Weekly Revenue Gap (%)', 
                     breaks = extended_breaks(n = 10), labels = percent) +
  theme(plot.title = element_text(size = 12, face = 'bold.italic', hjust = 0.5),
        axis.title.x = element_text(color = 'black', size = 11, face = 'bold'),
        axis.title.y = element_text(color = 'black', size = 11, face = 'bold'),
        panel.border = element_rect(colour = "black", fill = NA)) +
  facet_wrap( ~ scenario, ncol=2)

# Show the graphs
plt_HD
plt_LD


