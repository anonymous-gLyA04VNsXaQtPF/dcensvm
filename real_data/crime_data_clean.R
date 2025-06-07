#This script processes and cleans the Communities and Crime dataset (\url{https://archive.ics.uci.edu/dataset/183/communities+and+crime}) for analysis.

library(readxl)
library(ggplot2)
library(reshape2)


header <- read.csv("real_data/attributes.csv")
# trim whitespace from column names
header$attributes <- trimws(header$attributes)
df_raw <- read_excel("real_data/communities_and_crime.xlsx", na = c("", "NA", "NaN", "NULL", "?"))
colnames(df_raw) <- header$attributes
# save the raw data to a new file
write.csv(df_raw, "real_data/crime_raw.csv", row.names = FALSE)

dat_crime <- read.csv("real_data/crime_raw.csv", na = c("", "NA", "NaN", "NULL", "?"))


# Assume your dataframe is 'data'
columns_to_keep <- c('state', 'county', 'community', 'communityname', 'fold', 'ViolentCrimesPerPop')

# Identify columns with missing values, excluding columns we want to keep
cols_with_na <- sapply(dat_crime, function(x) any(is.na(x)))  # Find all columns with missing values
cols_to_drop <- names(dat_crime)[cols_with_na]               # Extract their names
cols_to_drop <- setdiff(cols_to_drop, columns_to_keep)       # Exclude specified columns to keep

# Drop columns with missing values, but keep specified ones
data <- dat_crime[, !(names(dat_crime) %in% cols_to_drop)]

# View the cleaned dataframe
head(data)

state <- as.factor(data$state)
unique(data$state)

# Create a mapping function to map state codes to regions
state_to_division <- function(state) {
  if (state %in% c(9, 23, 25, 33, 44, 50)) {
    return("New England")
  } else if (state %in% c(34, 36, 42)) {
    return("Middle Atlantic")
  } else if (state %in% c(17, 18, 26, 39, 55)) {
    return("East North Central")
  } else if (state %in% c(19, 20, 27, 29, 31, 38, 46)) {
    return("West North Central")
  } else if (state %in% c(10, 11, 12, 13, 24, 37, 45, 51, 54)) {
    return("South Atlantic")
  } else if (state %in% c(1, 21, 28, 47)) {
    return("East South Central")
  } else if (state %in% c(5, 22, 40, 48)) {
    return("West South Central")
  } else if (state %in% c(4, 8, 16, 30, 32, 35, 49, 56)) {
    return("Mountain")
  } else if (state %in% c(2, 6, 15, 41, 53)) {
    return("Pacific")
  } else {
    return(NA)  # Return NA if the state code is not recognized
  }
}

# Apply the function and add a new column 'division'
data$division <- sapply(data$state, state_to_division)
data$division <- as.factor(data$division)
levels(data$division)

# Convert ViolentCrimesPerPop into binary risk level based on median
median_value <- median(data$ViolentCrimesPerPop)
data$risk_level <- ifelse(data$ViolentCrimesPerPop > median_value, 1, 0)



# Count high-risk and low-risk samples for each division
risk_count <- aggregate(risk_level ~ division, data = data,
                        FUN = function(x) c(low = sum(x == 0), high = sum(x == 1)))
risk_count <- do.call(data.frame, risk_count)


# Reshape the dataframe for ggplot2 plotting
risk_count <- melt(risk_count, id.vars = "division",
                   variable.name = "risk", value.name = "count")

# Replace values in 'risk' column with "Low" and "High"
risk_count$risk <- ifelse(risk_count$risk == "risk_level.low", "Low", "High")


# Plot frequency bar chart
ggplot(risk_count, aes(x = division, y = count, fill = risk)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Division", y = "Count") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# Create a mapping table for divisions to numeric labels
division_mapping <- c(
  "New England" = 1,
  "Middle Atlantic" = 2,
  "East North Central" = 3,
  "West North Central" = 4,
  "South Atlantic" = 5,
  "East South Central" = 6,
  "West South Central" = 7,
  "Mountain" = 8,
  "Pacific" = 9
)

# Convert 'division' names to numeric labels
data$division_label <- as.numeric(division_mapping[data$division])

# View the result
table(data$division_label)

# Export the processed data to Excel
# write.xlsx(data, "real_data/crime_cleaned.xlsx")
# Save the cleaned data to a new CSV file
write.csv(data, "real_data/crime_cleaned.csv", row.names = FALSE)
