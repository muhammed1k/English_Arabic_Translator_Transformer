import tensorflow as tf
import pathlib
from config import config
import schedulers
from models import *
import data
import loss
import accuracy
from train import train_loop

def load_data(path_to_data):
    path_to_file = pathlib.Path(path_to_data).parent/'ara-eng/ara_eng.txt'
    data = path_to_file.read_text(encoding='utf-8')
    data = [line.split('\t') for line in data.splitlines()]
    en,ar = zip(*data)
    en = tf.constant(en)
    ar = tf.constant(ar)
    dataset = tf.data.Dataset.from_tensor_slices((en,ar)).shuffle(buffer_size=10000)
    
    return dataset,en,ar

def load_model(path_to_data,output_path,epochs,train=False):
    if path_to_data is None:
        path_to_data = config.path_to_data
    dataset,en,ar = load_data(path_to_data)
    learning_rate = schedulers.CustomSchedule(config.d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)
    train_batches,val_batches = data.load_batches_for_train(dataset)

    transformer = Transformer(
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        dff=config.dff,
        input_vocab_size=data.tokenizers.en.get_vocab_size().numpy(),
        target_vocab_size=data.tokenizers.ar.get_vocab_size().numpy(),
        dropout_rate=config.dropout_rate)

    for (en, ar), ar_labels in train_batches.take(1):
        break
    # to build the model
    build_model = transformer((en, ar))

    transformer.compile(
        loss=loss.masked_loss,
        optimizer=optimizer,
        metrics=[accuracy.masked_accuracy])

    transformer.load_weights(config.saved_weights)

    if train:
        transformer = train_loop(transformer,train_batches,val_batches,output_path,epochs)

    return transformer
