'''
This script is mostly the same as classification_evaluation.py but with modifications to 
create the csv file needed for submission on HuggingFace for accuracy evaluation.

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
    # With help from Copilot prompt: "get label and logits from the model output"
    
    batch_size = model_input.shape[0]
    # Initialize the best loss and best answer for each sample in the batch
    best_loss = [float('inf')] * batch_size  # Start with infinity for comparison
    best_label = [0] * batch_size  # Default answer is 0 for all samples
    
    # # Initialize a tensor to store logits for each class (4 classes in this case)
    logits = torch.zeros((4, batch_size), device=device)

    for i in range(4):
        # Compute the loss for the current class using the model's output
        curr_loss = discretized_mix_logistic_loss(model_input, model(model_input, [i] * batch_size), sum_batch=False)
        # Store the computed loss in the logits tensor
        logits[i] = curr_loss
        # Update the best loss and best answer for each sample in the batch
        for j in range(batch_size):
            if curr_loss[j] < best_loss[j]:  # Check if the current loss is better
                best_label[j] = i  # Update the best answer to the current class
                best_loss[j] = curr_loss[j]  # Update the best loss
    
    # Compute the column-wise sum of logits
    column_sums = torch.sum(logits, dim=0)
    # Switch the logits to represent probabilities (1 - normalized loss)
    switched_logits = 1 - logits / column_sums
    # Compute the new column-wise sum of switched logits
    new_column_sums = torch.sum(switched_logits, dim=0)
    # Normalize the switched logits to ensure they sum to 1
    normalized_logits = switched_logits / new_column_sums

    loss = torch.tensor(best_loss, device=device).detach()
    label = torch.tensor(best_label, device=device).detach()
    norm_logits = torch.tensor(normalized_logits, device=device).detach()
    return loss, label, norm_logits
# End of your code

# Modifying the classifier function to save the labels to a CSV file for Huggingface submission
def classifier(model, data_loader, dataset, device):
    model.eval()
    all_labels = []
    with torch.no_grad():
        for batch_idx, item in enumerate(tqdm(data_loader)):
            model_input, categories = item
            model_input = model_input.to(device)
            _, answer, _ = get_label(model, model_input, device)
            # logits_all.append(logits.T.cpu())
            all_labels.append(answer.cpu())

    # Generated using Copilot prompt: "save all_labels list to csv"
    save_answers = np.concatenate(all_labels, axis=0)
    with open("hf_submission.csv", mode='w', newline='') as file:
        writer = csv.writer (file)
        for image_path, answer in zip(dataset.samples, save_answers):
            img_name = os.path.basename(image_path[0])
            writer.writerow([img_name, answer])
        print("Labels saved to hf_submission.csv")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset') # changed to test for submission
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=False, # changed to False for accuracy submission
                                             **kwargs)

    #TODO:Begin of your code
    #You should replace the random classifier with your trained model
    model = PixelCNN(nr_resnet=2, nr_filters=100,
                     input_channels=3, nr_logistic_mix=5)
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model_path = os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()

    # store dataset for passing to classifi er function
    dataset = CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms)
    classifier(model=model, data_loader=dataloader, dataset=dataset, device=device)
    print('csv file saved')
        
        