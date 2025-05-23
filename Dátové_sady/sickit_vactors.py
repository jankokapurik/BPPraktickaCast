import pandas as pd
import tensorflow as tf

# loading raw dataset
raw_data = pd.read_csv('sources/candidates_raw.csv')


documentAssembler = documentAssembler().setInputCol("text").setOutputCol("document")