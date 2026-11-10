import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm



#load csv
df=pd.read_csv('data.csv')

# Select the 'price' column as your target variable (y)
y = df[['price']]

# Select the simple regression feature.
regression_feature= df[['sqft_living']]

#Select the nine multiple regression features .
multiple_regression_features=df[['bedrooms',
                                'bathrooms',
                                'sqft_living',
                                'sqft_lot',
                                'floors',
                                'waterfront',
                                'view',
                                'condition',
                                'yr_built']]

#print(y.shape)
#print(regression_feature.shape)
#print(multiple_regression_features.shape)
print("-"*50)
print("Splitting Dataset For Simple LinearRegression Model")
simple_X_train,simple_X_test,simple_y_train,simple_y_test = train_test_split(
    regression_feature, y, test_size=0.20, random_state=42, shuffle=True
)
print("Splitting Dataset For Simple LinearRegression Model Completed")
print("-"*50)

print("-"*50)
print("Splitting Dataset For Multiple LinearRegression Model")
multiple_X_train,multiple_X_test,multiple_y_train,multiple_y_test = train_test_split(
    multiple_regression_features, y, test_size=0.20, random_state=42, shuffle=True
)
print("Splitting Dataset For Multiple LinearRegression Model Completed")
print("-"*50)

print("-"*50)
print("Tranning Simple LinearRegression Model")
# Define and Initialize the LinearRegression Model
simple_model = LinearRegression()
simple_model.fit(simple_X_train, simple_y_train)
simple_prediction = simple_model.predict(simple_X_test)
print("Tranning Simple LinearRegression Model Completed")
print("-"*50)


print("-"*50)
print("Calculating mean_absolute_error for Simple LinearRegression Model")


simple_mae = mean_absolute_error(simple_y_test, simple_prediction)
print(f"Mean Absolute Error Of Simple_Model: {simple_mae}")


print("Calculating mean_absolute_error for Simple LinearRegression Model Completed")
print("-"*50)

print("-"*50)
print("Calculating Mean Squared Error for Simple LinearRegression Model")

simple_mse = mean_squared_error(simple_y_test, simple_prediction)
print(f"Mean Squared Error: {simple_mse}")


print("Calculating Mean Squared Error for Simple LinearRegression Model Completed")
print("-"*50)


print("-"*50)
print("Calculating Model explanatory performance(R²) for Simple LinearRegression Model")


simple_r2 = r2_score(simple_y_test, simple_prediction)
print(f"R-squared: {simple_r2}")


print("Calculating Model explanatory performance(R²) for Simple LinearRegression Model Completed")
print("-"*50)


# Generate the plot
plt.scatter(regression_feature, y, color="green", marker="o", label="Data Points")
plt.plot(simple_X_test, simple_prediction, color="red", linewidth=2, label="Regression Line")
# Add details
plt.title("")
plt.xlabel("sqft_living")
plt.ylabel("Price")
plt.legend()

# Display the chart
#plt.show()
simple_coefficient = simple_model.coef_
print(f"Simple Model Coefficient: {simple_coefficient}")

simple_intercept = simple_model.intercept_
print(f"Simple Model Intercept: {simple_intercept}")


print("-"*50)
print("Tranning Multiple LinearRegression Model")


# Define and Initialize the multiple_regression Model
multiple_model = LinearRegression()
multiple_model.fit(multiple_X_train, multiple_y_train)
multiple_prediction = multiple_model.predict(multiple_X_test)


print("Tranning Multiple LinearRegression Model Completed")
print("-"*50)


print("-"*50)
print("Calculating mean_absolute_error for Multiple LinearRegression Model")


multiple_mae = mean_absolute_error(multiple_y_test, multiple_prediction)
print(f"Mean Absolute Error Of Multiple_Model: {multiple_mae}")


print("Calculating mean_absolute_error for Multiple LinearRegression Model Completed")
print("-"*50)


print("-"*50)
print("Calculating Mean Squared Error for Multiple LinearRegression Model")

multiple_mse = mean_squared_error(multiple_y_test, multiple_prediction)
print(f"Mean Squared Error: {multiple_mse}")


print("Calculating Mean Squared Error for Multiple LinearRegression Model Completed")
print("-"*50)


print("-"*50)
print("Calculating Model explanatory performance(R²) for multiple LinearRegression Model")


multiple_r2 = r2_score(multiple_y_test, multiple_prediction)
print(f"R-squared: {multiple_r2}")


print("Calculating Model explanatory performance(R²) for multiple LinearRegression Model Completed")
print("-"*50)


# Generate the plot
plt.scatter(multiple_y_test, multiple_prediction, color="blue", marker="o", label="Data Points")
# Reference diagonal: Predicted Price = Actual Price
line_min = min(multiple_y_test['price'].min(), multiple_prediction.min())
line_max = max(multiple_y_test['price'].max(), multiple_prediction.max())

plt.plot([line_min, line_max],
         [line_min, line_max],
         color="red",
         linewidth=2,
         label="Perfect Prediction")


# Add details
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim(line_min, line_max)
plt.ylim(line_min, line_max)
plt.legend()

# Display the chart
#plt.show()

multiple_intercept = multiple_model.intercept_
print(f"Multiple Model Intercept: {multiple_intercept}")

multiple_coefficients = multiple_model.coef_
print(f"Multiple Model Coefficients: {multiple_coefficients}")

