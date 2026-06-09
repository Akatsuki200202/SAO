import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
basic_path="./checkpoints/PubMed_disjoint/clients_10/20250616_100608_Krum_none/"
chosen_list = np.load(basic_path+"chosen_list.npy")
print(Counter(chosen_list))
epochs = range(100)
plt.figure(figsize=(12, 6))
plt.scatter(epochs, chosen_list, label="chosen_client", color="green")
# plt.plot(epochs, asr_list[len(asr_list)-epoch :len(asr_list)], label="asr", color="red")
# plt.plot(epochs, dis_list, label="distance", color="green")
plt.xlabel("Epochs")
plt.ylabel("Client")
plt.title("Chosen Client in every epoch")
plt.legend()
plt.show()