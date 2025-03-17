using MLJ

# Load iris dataset
iris = MLJ.dataset("iris")
select!(iris, Not(:species))  # Remove the target column for feature extraction

# Define input features and target
X, y = iris[:, 1:2], iris[:, 3]

# Load and instantiate a logistic regression model
model = @load LogisticClassifier pkg=MLJLinearModels
logistic = model()

# Wrap data into an MLJ-compatible format
train, test = partition(eachindex(y), 0.8)  # 80-20 train-test split
X_train, y_train = X[train, :], y[train]
X_test, y_test = X[test, :], y[test]

# Train the model
mach = machine(logistic, X_train, y_train) |> fit!

# Make predictions
y_pred = predict_mode(mach, X_test)

# Evaluate accuracy
accuracy = sum(y_pred .== y_test) / length(y_test)
println("Logistic Regression Test Accuracy: $accuracy")
