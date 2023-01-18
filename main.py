import torch
from torch import nn, optim
from tqdm import tqdm
from dataset.mnist import get_train_dataloader, get_test_dataloader
from model.resnet import get_resnet
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#ネットワークをGPUに転送
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_resnet(pretrained=True).to(device)
#データローダー
dataloader1 = get_train_dataloader(root="data", batch_size=64)
dataloader2 = get_test_dataloader(root='data', batch_size=64)
# オプティマイザーの定義
optimizer = optim.SGD(params=model.parameters(),lr=1e-1)
# 損失関数の定義
criterion = nn.CrossEntropyLoss()
#epoch数
total_epoch = 20

#学習部分の関数
def train(epochs):
    train_accuracy, train_loss = 0.0, 0.0
    model.train()
    print('\nTrain start')
    for images, labels in tqdm(dataloader1):
        #viewで1次元配列に変更
        images, labels = images.to(device), labels.to(device)
        #勾配をリセット
        optimizer.zero_grad()
        # モデルからの出力
        out = model(images)
        # lossの算出
        loss = criterion(out, labels)
        #誤差逆伝播
        loss.backward()
        #パラメータ更新
        optimizer.step()
        #lossを更新
        train_loss += loss.item()
        # 推測値
        preds = out.argmax(axis=1)
        # 正答率の算出
        train_accuracy += torch.sum(preds == labels).item() / len(labels)
     #値の出力
    print(f"epoch: {epochs + 1}")
    print(f"train_loss: {train_loss / len(dataloader1)}")
    print(f"train_accuracy: {train_accuracy / len(dataloader1)}")
    with open('./result/data20_train.csv', 'a') as f:
            f.write('{:<3d},{:<3f},{:<3f}\n'.format(epochs+1,train_loss / len(dataloader1),train_accuracy / len(dataloader1)))
    train_loss = train_loss / len(dataloader1)
    train_accuracy = train_accuracy / len(dataloader1)
    return train_loss, train_accuracy

#検証部分
def val(epochs):
    val_accuracy, val_loss = 0.0, 0.0   
    preds_list = []
    true_list = [] 
    data_list = []
    model.eval()
    print('\nValidation start')
    with torch.no_grad():
        for images, labels in tqdm(dataloader2):
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            val_loss += loss.item()
            preds = out.argmax(axis=1)
            preds_list += preds.detach().cpu().numpy().tolist()
            true_list += labels.detach().cpu().numpy().tolist()
            data_list.append(images.cpu())
            val_accuracy += torch.sum(preds == labels).item() / len(labels)
     #値の出力
    print(f"epoch: {epochs + 1}")
    print(f"Validation loss: {val_loss / len(dataloader2)}")
    print(f"Validation accuracy: {val_accuracy / len(dataloader2)}")
    with open('./result/data20_val.csv', 'a') as f:
            f.write('{:<3d},{:<3f},{:<3f}\n'.format(epochs+1,val_loss / len(dataloader2),val_accuracy / len(dataloader2)))
    val_loss = val_loss / len(dataloader2)
    val_accuracy = val_accuracy / len(dataloader2)
    return true_list, preds_list, data_list, val_loss, val_accuracy
#実行部分
def run():
    #リスト作成
    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    
    for epochs in range(total_epoch):
        train_loss, train_accuracy =train(epochs)
        true_list, preds_list, data_list, val_loss, val_accuracy = val(epochs)
        #リストに追加
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)
    #混同行列作成
    cm = confusion_matrix(true_list, preds_list)
    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predict class', fontsize=13)
    plt.ylabel('True class', fontsize=13)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig('./result/confusion_matrix_15.png')
    #lossグラフ作成
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train')
    ax.plot(range(len(val_loss_list)), val_loss_list, c='r', label='validation')
    ax.set_xlabel('epoch', fontsize='20')
    ax.set_ylabel('loss', fontsize='20')
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6])
    ax.set_title('train and validation loss', fontsize='20')
    ax.grid()
    ax.legend(fontsize='20')
    plt.show()
    plt.savefig('./result/loss_graph_20.png')
     #accuracyグラフ作成
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(train_accuracy_list)), train_accuracy_list, c='b', label='train')
    ax.plot(range(len(val_accuracy_list)), val_accuracy_list, c='r', label='validation')
    ax.set_xlabel('epoch', fontsize='20')
    ax.set_ylabel('accuracy', fontsize='20')
    ax.set_yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ax.set_title('train and validation accuracy', fontsize='20')
    ax.grid()
    ax.legend(fontsize='20')
    plt.show()
    plt.savefig('./result/accuracy_graph_20.png')
    #間違えた数字の可視化
    fig = plt.figure(figsize=(20,5))
    data_block = torch.cat(data_list,dim=0)
    idx_list = [n for n,(x,y) in enumerate(zip(true_list,preds_list)) if x!=y]
    len(idx_list)
    for i,idx in enumerate(idx_list[:20]):
        ax = fig.add_subplot(2,10,1+i)
        ax.axis('off')
        ax.set_title(f'true:{true_list[idx]} pred:{preds_list[idx]}')
        ax.imshow(data_block[idx,0])
        plt.savefig('./result/misspreddata_20')

if __name__ == "__main__":
    run()