import pandas as pd
from matplotlib import pyplot as plt

path = "D://datasets//iris//Loss_Restricted_BNN.xlsx"

data = pd.read_excel(path)

plt.figure()
plt.plot(data['RBNN'])
plt.plot(data['FFNN'])
plt.plot(data['BNN'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epochs vs Error plots')
plt.legend(['RBNN','FFNN','BNN'])
plt.show()
plt.savefig('comparison_loss.pdf')
