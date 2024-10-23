
def train_loop(transformer,train_batches,val_batches,output_path,epochs):
    if epochs is None:
        epochs = 20
    if output_path is None:
        output_path = r'weights/saved_weights.h5'
    transformer.fit(train_batches,
                    epochs=epochs,
                    validation_data=val_batches)
    
    transformer.save_weights(output_path)

    return transformer