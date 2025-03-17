using Flux, Flux.Data.MNIST, Images, Statistics

# Load MNIST dataset
train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

# Preprocess data: Flatten 28x28 images to 784x1 vectors
X_train = reshape(train_x, 28*28, :)
X_test = reshape(test_x, 28*28, :)

# Convert labels to one-hot encoding
Y_train = Flux.onehotbatch(train_y, 0:9)
Y_test = Flux.onehotbatch(test_y, 0:9)

# Define a neural network model
model = Chain(
    Dense(28*28, 128, relu),  # Hidden layer
    Dense(128, 64, relu),     # Another hidden layer
    Dense(64, 10),            # Output layer (10 classes)
    softmax
)

# Set up training
loss(x, y) = Flux.crossentropy(model(x), y)
optimizer = Adam()  # Adaptive gradient descent

# Train the model
Flux.train!(loss, Flux.params(model), [(X_train, Y_train)], optimizer)

# Test accuracy
accuracy = sum(argmax.(eachcol(model(X_test))) .== test_y) / length(test_y)
println("Test Accuracy: $accuracy")
