import matplotlib.pyplot as plt
import pandas as pd

csv_path1 = r'mnist/result/data17_train.csv'
csv_path2 = r'mnist/result/data17_val.csv'
csv_df1 = pd.read_csv(csv_path1, header=None)
csv_df2 = pd.read_csv(csv_path2, header=None)
csv_epoch = pd.read_csv(csv_path1)
data1_loss = csv_df1[csv_df1.columns[1]]
data2_loss = csv_df2[csv_df2.columns[1]]
data1_accuracy = csv_df1[csv_df1.columns[2]]
data2_accuracy = csv_df2[csv_df2.columns[2]]
data_epoch = csv_epoch[csv_epoch.columns[0]]

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(data1_loss, c= 'b', label='train')
ax.plot(data2_loss, c= 'r', label='validation')

ax.set_xlabel('epoch', fontsize='25')
ax.set_xticks([0,4,8,12,16,20])
ax.set_ylabel('loss', fontsize='25')
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
ax.grid()
ax.legend(fontsize='25')
ax.tick_params(labelsize=20)
#plt.show()
plt.savefig('loss_graph_correct17.png')


fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(data1_accuracy, c= 'b', label='train')
ax.plot(data2_accuracy, c= 'r', label='validation')

ax.set_xlabel('epoch', fontsize='25')
ax.set_xticks([0,4,8,12,16,20])
ax.set_ylabel('accuracy', fontsize='25')
ax.set_yticks([0.7,0.8,0.9,1.0])
ax.grid()
ax.legend(fontsize='25')
ax.tick_params(labelsize=20)
#plt.show()
plt.savefig('accuracy_graph_correct17.png')