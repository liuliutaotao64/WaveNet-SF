
import os
import json
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from utils import read_test_data, MyDataSet
from model import WaveNet_SF
from sklearn.metrics import roc_auc_score 


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)
        return acc

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)


        plt.xticks(range(self.num_classes), self.labels, rotation=45)

        plt.yticks(range(self.num_classes), self.labels)

        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')


        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):

                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def test(test_data_path,model_weights,last_10_epoch=True):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    test_data_path = test_data_path

    test_images_path, test_images_label = read_test_data(test_data_path)

    data_transform = transforms.Compose([transforms.Resize(456),
                                         transforms.CenterCrop(448),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.21, 0.21, 0.21], [0.16, 0.16, 0.16])])
    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=32,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4,
                                              collate_fn=test_dataset.collate_fn)

    net = WaveNet_SF()

    model_weights = model_weights
    model_weights_path_list = []

    model_weights_path_list.append(os.path.join(model_weights,'model_best.pth'))

    test_txt_dir = model_weights
    if not os.path.exists(test_txt_dir):
        os.makedirs(test_txt_dir)

    test_path = os.path.join(test_txt_dir,'test.txt')

    for model_weight_path in model_weights_path_list:
        assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
        net.to(device)

        # read class_indict
        json_label_path = './class_indices.json'
        assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
        json_file = open(json_label_path, 'r')
        class_indict = json.load(json_file)

        labels = [label for _, label in class_indict.items()]
        confusion = ConfusionMatrix(num_classes=8, labels=labels)
        all_labels = []
        all_preds = []
        net.eval()
        with torch.no_grad():
            for val_data in tqdm(test_loader):
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))

                outputs = torch.softmax(outputs, dim=1)
                all_preds.extend(outputs.cpu().numpy())
                outputs = torch.argmax(outputs, dim=1)
                confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
                all_labels.extend(val_labels.cpu().numpy())  #

        all_preds = np.vstack(all_preds)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
        print(f"AUC: {auc}")  #
        confusion.plot()
        accuracy = confusion.summary()

        with open(test_path,'a',encoding='utf-8') as f:
            f.write(f"\n{'-'*20}\n")
            f.write(f"Model: {os.path.basename(model_weight_path)}\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"AUC: {auc}\n")
            f.write("\nConfusion Matrix:\n")
            np.savetxt(f, confusion.matrix, fmt='%d', delimiter='\t', header='\t'.join(labels))
            f.write("\n")
            print(f"Results for {os.path.basename(model_weight_path)} saved to {test_path}")
            confusion.plot()

if __name__ == '__main__':
    test('',model_weights='',)







