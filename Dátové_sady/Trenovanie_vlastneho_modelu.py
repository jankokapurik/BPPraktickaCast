
import fasttext

model = fasttext.train_unsupervised(
    input="Dátové_sady/wiki_sentences.txt",
    model='skipgram',
    dim=300,
    epoch=10,
    minCount=5,
    thread=4
)

model.save_model("Models/custom_model.bin")