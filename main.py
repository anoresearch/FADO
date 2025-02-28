from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from FADO import FADO
from data_loader import load_and_preprocess_data

# Load and preprocess data
class_a, class_b = 0, 1  # Example: Use CIFAR-10 class 0 and class 1
x_train, y_train, x_test, y_test = load_and_preprocess_data(class_a, class_b)

# Define CNN model
def create_cnn_model():
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

# Initialize FADO
fado = FADO(alpha=1.0, beta=0.9, gamma=0.001, update_interval=5)
n_majority, n_minority = np.sum(y_train == 0), np.sum(y_train == 1)
w0, w1 = fado.initialize_weights(n_majority, n_minority)

# Compile model
model = create_cnn_model()
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Training loop
batch_size = 64
epochs = 10
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        y_pred = model.predict(x_batch, verbose=0).flatten()
        if epoch % fado.update_interval == 0:
            bias = fado.compute_bias(y_batch, y_pred)
            w0, w1 = fado.update_weights(bias)
        sample_weights = np.where(y_batch == 1, w1, w0)
        model.train_on_batch(x_batch, y_batch, sample_weight=sample_weights)
    val_loss, val_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Epoch {epoch + 1}: val_loss = {val_loss:.4f}, val_accuracy = {val_accuracy:.4f}")
    early_stopping.on_epoch_end(epoch, logs={'val_loss': val_loss})
    model_checkpoint.on_epoch_end(epoch, logs={'val_loss': val_loss})
    if early_stopping.stopped_epoch > 0:
        break

# Final evaluation
model.load_weights('best_model.h5')
final_loss, final_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Final Test Loss: {final_loss:.4f}, Final Test Accuracy: {final_accuracy:.4f}")
