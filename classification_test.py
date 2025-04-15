'''
This code is used to generate a csv file for the test set for leaderboard submission. Its mostly the same as classification_evaluation.py, 
but I modified the classifier function to get the labels for each image for csv submission and modifications to the original script wasn't allowed.
'''

from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

#TODO: Begin of your code
def get_label(model, model_input, device):
    num_classes = 4   
    batch_size = model_input.size(0)
    log_likelihood = torch.zeros(batch_size, num_classes, device=device)
    
    for c in range(num_classes):
        labels = torch.full((batch_size,), c, dtype=torch.long, device=device)
        model_output = model(model_input, labels)
        nll = discretized_mix_logistic_classify(model_input, model_output) #I modified loss to give output for each picture
        log_likelihood[:, c] = -nll  #negative log likelihood

    # I'm now selecting the class with the highest log likelihood==>(lowest nll)
    _, predicted_labels = log_likelihood.max(1)
    return predicted_labels
# End of your code

# I'm modifying the classifier function to get the labels for each image for csv submission
# I made a new file because modifying the original classifier function was not alowed
def classifier(model, data_loader, device, dataset):
    model.eval()
    preds = []

    # enumerate the data_loader and get the model input and labels for each batch
    for _, item in enumerate(tqdm(data_loader)):
        model_input, _ = item
        model_input = model_input.to(device)
        pred = get_label(model, model_input, device)
        preds.append(pred)
    preds = torch.cat(preds, -1)

    # write the final result csv file
    with open("test_submission.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        for path, label in zip(dataset.samples, preds):
            name = path[0][path[0].rfind('/')+1:] 
            writer.writerow([name, label.item()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=False, # do not shuffle for test set
                                             **kwargs)

    #TODO:Begin of your code
    #You should replace the random classifier with your trained model
    model = PixelCNN(nr_resnet=1, nr_filters=40, nr_logistic_mix=5, input_channels=3, num_classes=4)
    # model.load_state_dict(torch.load('models/conditional_pixelcnn_v2.pth'))
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model_path = os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn_v2.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()

    dataset = CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms)
    classifier(model = model, data_loader = dataloader, device = device, dataset=dataset)
    print('CSV file generated')