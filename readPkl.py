import pickle

with open("LH_CD_MATHEUS_SEIDEL.pkl", "rb") as arq:
    file = pickle.load(arq)
print(file)