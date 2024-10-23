class Config():
    def __init__(self):
        self.path_to_data = r"./ara-eng"
        self.model_name = 'tokenizer_translate_en_ar_converter'
        self.MAX_TOKENS=128
        self.BUFFER_SIZE = 20000
        self.BATCH_SIZE = 16
        self.num_layers = 2
        self.d_model = 128
        self.dff = 128
        self.num_heads = 4
        self.dropout_rate = 0.1
        self.saved_weights = r'./weights/saved_weights.h5'

config = Config()
