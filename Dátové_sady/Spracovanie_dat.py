import pandas as pd

candidates_df = pd.read_csv('Sources\candidates.txt')
candidates_df.columns = ['Word']

neologisms_df = pd.read_csv('Sources/NEOLOGISMS.txt')
neologisms_df.columns =  ['Word']
neologisms_set = set(neologisms_df['Word'])

def contains_neologism(word, neologisms):
    return any(neo in word for neo in neologisms)

candidates_df['Target'] = candidates_df['Word'].apply(lambda w: 1 if contains_neologism(w, neologisms_set) else 0)

candidates_df = candidates_df.sort_values(by='Word').reset_index(drop=True)

candidates_df.to_csv('Dátové_sady/candidates_dataset.csv', index=False)

print("Sum of labels (Target = 1):", candidates_df['Target'].sum())
