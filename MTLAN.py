try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
except ModuleNotFoundError as e:
    raise ImportError("TensorFlow is not installed or not accessible in this environment. Please install TensorFlow 2.x using `pip install tensorflow`.") from e

# Ensure TensorFlow 2.3.1 is used
assert tf.__version__ == '2.3.1', "Please install TensorFlow 2.3.1 to run this script."

# Define the MTLAN model
class MTLAN(Model):
    def __init__(self, hidden_units, k, l, w_tc, w_sc, output_dim, use_batch_norm=True, dropout_rate=0.3):
        """
        Multi-Task Learning Attention Network (MTLAN) model.

        Parameters:
        hidden_units (int): Number of units in the GRU layer.
        k (int): Number of filters for the transverse attention block.
        l (int): Number of filters for the longitudinal attention block.
        w_tc (int): Transverse convolution filter size.
        w_sc (int): Longitudinal convolution filter size.
        output_dim (int): Dimension of the output layer.
        use_batch_norm (bool): Whether to apply batch normalization in the model.
        dropout_rate (float): Dropout rate for regularization.
        """
        super(MTLAN, self).__init__()
        self.hidden_units = hidden_units
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # GRU for hidden state generation
        self.gru = layers.GRU(hidden_units, return_sequences=True)
        if self.use_batch_norm:
            self.batch_norm_gru = layers.BatchNormalization()
        self.dropout_gru = layers.Dropout(dropout_rate)

        # Transverse attention block (1D Convolution)
        self.transverse_conv = layers.Conv1D(k, w_tc, strides=1, activation='relu')
        if self.use_batch_norm:
            self.batch_norm_tc = layers.BatchNormalization()
        self.transverse_weights = layers.Dense(k, activation='softmax')

        # Longitudinal attention block (1D Convolution)
        self.longitudinal_conv = layers.Conv1D(l, w_sc, strides=1, activation='relu')
        if self.use_batch_norm:
            self.batch_norm_sc = layers.BatchNormalization()
        self.longitudinal_weights = layers.Dense(l, activation='softmax')

        # Information aggregation weights
        self.w_ht = layers.Dense(hidden_units)
        self.w_tc = layers.Dense(hidden_units)
        self.w_sc = layers.Dense(hidden_units)

        # Final prediction layer (multi-task output capability)
        self.output_layers = [layers.Dense(output_dim, activation='linear') for _ in range(2)]

    def call(self, x):
        """
        Forward pass for the MTLAN model.

        Parameters:
        x (tf.Tensor): Input tensor with shape (batch_size, time_steps, features).

        Returns:
        list of tf.Tensor: Output predictions for multiple tasks.
        """
        # Hidden state generation
        h = self.gru(x)
        if self.use_batch_norm:
            h = self.batch_norm_gru(h)
        h = self.dropout_gru(h)

        # Transverse attention block
        h_tc = self.transverse_conv(h)
        if self.use_batch_norm:
            h_tc = self.batch_norm_tc(h_tc)
        alpha = tf.nn.softmax(self.transverse_weights(h_tc), axis=-1)
        alpha_expanded = tf.expand_dims(alpha, axis=-1)  # [?, 1436, 32, 1]
        h_tc_expanded = tf.expand_dims(h_tc, axis=-1)    # [?, 1436, 32, 1]
        v_tc = tf.reduce_sum(alpha_expanded * h_tc_expanded, axis=1)

        # Longitudinal attention block
        h_t = tf.transpose(h, perm=[0, 2, 1])  # Transpose for column-wise operation
        h_sc = self.longitudinal_conv(h_t)
        if self.use_batch_norm:
            h_sc = self.batch_norm_sc(h_sc)
        beta = tf.nn.softmax(self.longitudinal_weights(h_sc), axis=-1)
        beta_expanded = tf.expand_dims(beta, axis=-1)    # [?, 1436, 32, 1]
        h_sc_expanded = tf.expand_dims(h_sc, axis=-1)    # [?, 1436, 32, 1]
        v_sc = tf.reduce_sum(beta_expanded * h_sc_expanded, axis=1)

        # Information aggregation
        v_tc = tf.squeeze(v_tc, axis=-1)  # Ensure shape [batch_size, hidden_units]
        v_sc = tf.squeeze(v_sc, axis=-1)  # Ensure shape [batch_size, hidden_units]

        h_t_new = self.w_ht(h[:, -1, :]) + self.w_tc(v_tc) + self.w_sc(v_sc)

        # Predictions for multiple tasks
        outputs = [output_layer(h_t_new) for output_layer in self.output_layers]
        return outputs

# Helper functions
def preprocess_data(data):
    """
    Preprocess the input data (normalization, missing data handling, etc.).

    Parameters:
    data (tf.Tensor): Raw data tensor.

    Returns:
    tf.Tensor: Normalized data tensor.
    """
    min_val = tf.reduce_min(data, axis=0)
    max_val = tf.reduce_max(data, axis=0)
    return (data - min_val) / (max_val - min_val)

def load_data():
    """
    Load and preprocess the dataset.

    Returns:
    tuple: Preprocessed training, validation, and test datasets and their labels.
    """
    train_data = tf.random.normal([100, 1440, 3])
    val_data = tf.random.normal([20, 1440, 3])
    test_data = tf.random.normal([20, 1440, 3])

    train_labels_task1 = tf.random.normal([100, 1])
    train_labels_task2 = tf.random.normal([100, 1])

    val_labels_task1 = tf.random.normal([20, 1])
    val_labels_task2 = tf.random.normal([20, 1])

    test_labels_task1 = tf.random.normal([20, 1])
    test_labels_task2 = tf.random.normal([20, 1])

    return (preprocess_data(train_data), (train_labels_task1, train_labels_task2),
            preprocess_data(val_data), (val_labels_task1, val_labels_task2),
            preprocess_data(test_data), (test_labels_task1, test_labels_task2))

def train_model(model, train_data, train_labels, val_data, val_labels, epochs=100, batch_size=16):
    """
    Train the MTLAN model.

    Parameters:
    model (MTLAN): The MTLAN model instance.
    train_data (tf.Tensor): Training data.
    train_labels (tuple of tf.Tensor): Training labels for each task.
    val_data (tf.Tensor): Validation data.
    val_labels (tuple of tf.Tensor): Validation labels for each task.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size.

    Returns:
    tf.keras.callbacks.History: Training history.
    """
    return model.fit(train_data, train_labels,
                     validation_data=(val_data, val_labels),
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=[
                         tf.keras.callbacks.EarlyStopping(
                             monitor='val_loss',
                             patience=15,
                             restore_best_weights=True
                         ),
                         tf.keras.callbacks.ModelCheckpoint(
                             filepath='best_mtlan_model',
                             save_best_only=True
                         ),
                         tf.keras.callbacks.TensorBoard(log_dir='./logs')
                     ])

def evaluate_model(model, test_data, test_labels):
    """
    Evaluate the trained model.

    Parameters:
    model (MTLAN): The trained MTLAN model instance.
    test_data (tf.Tensor): Test data.
    test_labels (tuple of tf.Tensor): Test labels for each task.

    Returns:
    dict: Test loss and mean absolute error (MAE) for each task.
    """
    results = model.evaluate(test_data, test_labels)
    return {
        "test_loss": results[0],
        "test_mae_task1": results[1],
        "test_mae_task2": results[2]
    }

# Main execution
if __name__ == "__main__":
    # Hyperparameters
    hidden_units = 256
    k = 32  # Number of transverse filters
    l = 32  # Number of longitudinal filters
    w_tc = 5  # Transverse filter size
    w_sc = 5  # Longitudinal filter size
    output_dim = 1  # Prediction output size
    use_batch_norm = True  # Enable or disable batch normalization
    dropout_rate = 0.3  # Dropout rate for regularization

    # Initialize the model
    mtlan_model = MTLAN(hidden_units, k, l, w_tc, w_sc, output_dim, use_batch_norm=use_batch_norm, dropout_rate=dropout_rate)

    # Compile the model
    mtlan_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss='mse',
                        metrics=['mae'])

    # Load and preprocess data
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data()

    # Train the model
    print("Starting training...")
    history = train_model(mtlan_model, train_data, train_labels, val_data, val_labels)

    # Evaluate the model
    print("Evaluating model...")
    test_results = evaluate_model(mtlan_model, test_data, test_labels)
    print(f"Test Results: {test_results}")

    # Save the model
    print("Saving model...")
    mtlan_model.save("mtlan_model", save_format="tf")
    print("Model saved in TensorFlow SavedModel format.")
