# Plot codes of Figure A.1 in Supplementary Material under slurm jobs in HPC
sim_name <- "sim_plot_types_of_loss"
##################### Different types of kernel ##################
library(ggplot2)
library(dplyr)
library(tidyr)

# Define variables and bandwidth
v <- seq(-1, 3, by = 0.01)
h <- 0.2

# Define various loss functions
hinge_loss <- function(v) pmax(0, 1 - v)
uniform_kernel_loss <- function(v, h) ifelse(v <= 1 - h, 1 - v,
                                             ifelse(v <= 1 + h, ((1 + h - v)^2) / (4 * h), 0))
laplacian_kernel_loss <- function(v, h) ifelse(v < 1,
                                               1 + (h / 2) * exp((v - 1) / h) - v,
                                               (h / 2) * exp((1 - v) / h))
logistic_kernel_loss <- function(v, h) -v + h * log(exp(1 / h) + exp(v / h))
gaussian_kernel_loss <- function(v, h) {
  phi <- pnorm((1 - v) / h)
  pdf <- dnorm((1 - v) / h)
  (1 - v) * phi + h * pdf
}
epanechnikov_kernel_loss <- function(v, h) ifelse(v <= 1 - h, 1 - v,
                                                  ifelse(v <= 1 + h,
                                                         ((1 - v + h)^3 * (3 * h - (1 - v))) / (16 * h^3), 0))

# Create a data frame
df <- data.frame(
  v = v,
  Hinge = hinge_loss(v),
  Uniform = uniform_kernel_loss(v, h),
  Laplacian = laplacian_kernel_loss(v, h),
  Logistic = logistic_kernel_loss(v, h),
  Gaussian = gaussian_kernel_loss(v, h),
  Epanechnikov = epanechnikov_kernel_loss(v, h)
)

# Convert to long format
df_long <- pivot_longer(df, -v, names_to = "Kernel", values_to = "Loss")

# Order the factor levels for the Kernel variable
df_long$Kernel <- factor(df_long$Kernel, levels = c(
  "Hinge", "Uniform", "Laplacian", "Logistic", "Gaussian", "Epanechnikov"
))

# Create the plot
p <- ggplot(df_long, aes(x = v, y = Loss, color = Kernel, linetype = Kernel)) +
  geom_line(size = 0.7) +
  labs(x = "", y = "") +
  # scale_color_manual(values = set1_colors) +
  theme_minimal(base_size = 14) +
  ggpubr::theme_pubr() +
  theme(
    legend.position = c(0.95, 0.95),  # Top-right corner inside the plot
    legend.justification = c("right", "top"),
    legend.title = element_blank(),
    legend.box.just = "right",
    legend.text = element_text(size = 12),
    legend.key.width = unit(2.5, "lines")  # Increase legend line length
  ) +
  guides(
    color = guide_legend(ncol = 1),
    linetype = guide_legend(ncol = 1)
  )

# Load ggpubr for color palette
library(ggpubr)

# Set the color palette to NPG (Nature Publishing Group)
set_palette(p, "npg")

# Save the plot
ggsave("Output/figs/hinge_loss_plot.pdf", plot = last_plot(),
       width = 6, height = 5, units = "in")


##################### Different values of h ##################
library(ggplot2)
library(dplyr)
library(tidyr)



# Define the input range
v <- seq(-2, 4, by = 0.01)
h_vals <- c(0.1, 0.2, 0.4, 0.6, 0.8)

# Generate data for Logistic smoothed loss
df_list <- lapply(h_vals, function(h) {
  data.frame(
    v = v,
    Loss = logistic_kernel_loss(v, h),
    h_label = paste0("h = ", h)
  )
})

# Add the original Hinge loss (treated as h = 0)
df_hinge <- data.frame(
  v = v,
  Loss = hinge_loss(v),
  h_label = "Hinge"
)

# Combine all data
df_all <- bind_rows(df_hinge, bind_rows(df_list))

# Order the factor levels for the h_label variable
df_all$h_label <- factor(df_all$h_label, levels = c(
  "Hinge", "h = 0.1", "h = 0.2", "h = 0.4", "h = 0.6", "h = 0.8"
))

# Create the plot
p <- ggplot(df_all, aes(x = v, y = Loss, color = h_label, linetype = h_label)) +
  geom_line(size = 0.7) +
  labs(x = "", y = "") +
  xlim(-2,4) +
  # scale_color_manual(values = set1_colors) +
  theme_minimal(base_size = 14) +
  ggpubr::theme_pubr() +
  theme(
    legend.position = c(0.95, 0.95),  # Top-right corner inside the plot
    legend.justification = c("right", "top"),
    legend.title = element_blank(),
    legend.box.just = "right",
    legend.text = element_text(size = 12),
    legend.key.width = unit(2.5, "lines")  # Increase legend line length
  ) +
  guides(
    color = guide_legend(ncol = 1),
    linetype = guide_legend(ncol = 1)
  )

# Load ggpubr for color palette
library(ggpubr)

# Set the color palette to NPG
set_palette(p, "npg")

# Save the plot
ggsave("Output/figs/loss_h_varied_plot.pdf", plot = last_plot(),
       width = 6, height = 5, units = "in")
