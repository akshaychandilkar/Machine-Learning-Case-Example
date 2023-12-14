import pandas as pd

print("-----Marvellous Infosystems by Piyush Khairnar-----")
print("Machine Learning Application")

# Step 1: Get Data
data = pd.read_csv("MarvellousAdvertising.csv")

# Step 2: Clean, Prepare, and Manipulate data
# In this case, we are assuming that the data doesn't require significant cleaning or manipulation.

# Step 3: Train Data using Custom Linear Regression

# Define a custom Linear Regression algorithm
def custom_linear_regression(X, y):
    n = len(X)
    sum_x = X.sum()
    sum_y = y.sum()
    sum_x_squared = (X ** 2).sum()
    sum_xy = (X * y).sum()

    # Calculate the coefficients (slope and intercept)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n

    return slope, intercept

# Split the data into X (features) and y (target)
X = data[["TV", "Radio", "Television"]]
y = data["Sales"]

# Convert X and y to numeric data types
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
X = X.dropna()
y = y.dropna()

# Call the custom Linear Regression function
slope, intercept = custom_linear_regression(X, y)

# Calculate R-squared
def calculate_r_squared(X, y, slope, intercept):
    y_mean = y.mean()
    y_predicted = X.apply(lambda x: slope * x + intercept, axis=1)

    ss_total = ((y - y_mean) ** 2).sum()
    ss_res = ((y - y_predicted) ** 2).sum()
    r_squared = 1 - (ss_res / ss_total)

    return r_squared

r_squared = calculate_r_squared(X, y, slope, intercept)

# Display the R-squared value
print(f"R-squared value: {r_squared}")


