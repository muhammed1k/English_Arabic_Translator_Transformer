import tensorflow as tf
import tensorflow_text as tf_text
from config import config

tokenizers = tf.saved_model.load(config.model_name)

def split_train_valid(dataset):
    total_size = sum(1 for _ in dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_data = dataset.take(train_size)
    validation_data = dataset.skip(train_size)

    return train_data,validation_data

def prepare_batch(en, ar):
    en = tokenizers.en.tokenize(en)      # Output is ragged.
    en = en[:, :config.MAX_TOKENS]    # Trim to MAX_TOKENS.
    en = en.to_tensor()  # Convert to 0-padded dense Tensor

    ar = tokenizers.ar.tokenize(ar)
    ar = ar[:, :(config.MAX_TOKENS+1)]
    ar_inputs = ar[:, :-1].to_tensor()  # Drop the [END] tokens
    ar_labels = ar[:, 1:].to_tensor()   # Drop the [START] tokens

    return (en, ar_inputs), ar_labels



def make_batches(ds):
    return (
        ds
        .shuffle(config.BUFFER_SIZE)
        .batch(config.BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))


def load_batches_for_train(dataset):
    train_data,validation_data = split_train_valid(dataset)
    # Create training and validation set batches.
    train_batches = make_batches(train_data)
    val_batches = make_batches(validation_data)
    return train_batches,val_batches
