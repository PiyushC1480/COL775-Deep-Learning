import torch
import torch.nn as nn
import random
import os
import pickle
import re
import json
import numpy as np

from nltk import word_tokenize
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers import BertModel
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"Seq2Seq with Attention using pre-trained frozen BERT-base-cased encoder and an LSTM decoder"

class ARGS():
    def __init__(self):
        self.model_type = "bert_lstm_attn_frozen"
        self.data_dir = "/home/cse/btech/cs1211010/a2/math-data"
        self.batch_size = 16
        self.num_workers = 4
        self.epochs = 30
        self.en_hidden = 512
        self.de_hidden = 512
        self.en_num_layers = 1
        self.de_num_layers = 1
        self.embed_dim = 300
        self.processed_data = "/home/cse/btech/cs1211010/a2/math-data"
        self.checkpoint_dir ="/home/cse/btech/cs1211010/a2/checkpoints"
        self.result_dir = "/home/cse/btech/cs1211010/a2/results"


def tokenize_problem(problem):
    return word_tokenize(problem)

def tokenize_formula(formula):
    all_formulas = formula.split("|")
    formula_split = []
    for s in all_formulas:
        s_brk = re.match(r"([a-zA-Z]+)\(([^,]+),([^)]+)\)", s)
        if s_brk:
            formula_split.extend([s_brk.group(1),"(", s_brk.group(2),",", s_brk.group(3), ")","|"])
    return formula_split

#DATASET
class TextToMathDataset(Dataset):
    def __init__(self,  file_path, data_path):
        self.file_path = file_path
        # pth = os.path.join(self.file_path, f"{data_prefix}.json")
        
        # print(pth)
        self.data  = json.load(open(data_path))
        print("Dataset Length =", len(self.data))

        with open(os.path.join(file_path, "encoder.vocab"), "r") as file:
            vocab = file.readlines()
        self.encoder_vocab = vocab
        
        with open(os.path.join(file_path, "decoder.vocab"), "r") as file:
            vocab = file.readlines()
        self.decoder_vocab = vocab

        with open(os.path.join(file_path, "encoder_word2idx.pickle"), "rb") as file:
            word2idx = pickle.load(file)
        with open(os.path.join(file_path, "encoder_idx2word.pickle"), "rb") as file:
            idx2word = pickle.load(file)

        self.en_word2idx = word2idx
        self.en_idx2word = idx2word

        with open(os.path.join(file_path, "decoder_word2idx.pickle"), "rb") as file:
            word2idx = pickle.load(file)
        with open(os.path.join(file_path, "decoder_idx2word.pickle"), "rb") as file:
            idx2word = pickle.load(file)
            
        self.de_word2idx = word2idx
        self.de_idx2word = idx2word
        self.en_tokenizer =  BertTokenizer.from_pretrained("bert-base-cased")


        print("Encoder Vocab Size = {}, Decoder Vocab Size = {}".format(len(self.en_word2idx), len(self.de_word2idx)))
        print("Encoder word2idx Size = {}, Decoder word2idx Size = {}".format(len(self.en_word2idx), len(self.de_word2idx)))

    def __len__(self):  
        return len(self.data)
    

    def __getitem__(self, index):
        problem = self.data[index]["Problem"]
        problem = self.en_tokenizer.encode(problem)

        linear_formula = self.data[index]["linear_formula"]
        formula_split = ["<sos>"] +tokenize_formula(linear_formula) + ["<eos>"]
        
        aa = self.data[index]["answer"]

        formula_split = [self.de_word2idx[q] if q in self.de_word2idx else self.de_word2idx["<unk>"] for q in formula_split]
        
        answer = {"Problem" : problem , "linear_formula" : formula_split, "answer" : aa }
        return answer
    

def collate(batch):
    max_len_problem = max([len(sample['Problem']) for sample in batch])

    problem_lens = torch.zeros(len(batch), dtype=torch.long)
    padded_problem = torch.zeros((len(batch), max_len_problem), dtype=torch.long)
    problem_attn_mask = torch.zeros((len(batch), max_len_problem), dtype=torch.long)
    for idx in range(len(batch)):
        problem = batch[idx]['Problem']
        prob_len = len(problem)
        
        problem_lens[idx] = prob_len
        
        problem_tensor = torch.LongTensor(problem)
        
        problem_attn_mask[idx, :prob_len] = torch.ones((1, prob_len), dtype=torch.long)
        
    ret = {'problem': padded_problem, 'problem_attn_mask' : problem_attn_mask, 'problem_lens': problem_lens}

    return ret




# MODEL
SPL_TOKEN = ["<pad>", "<unk>", "<sos>", "<eos>"]
    

#LSTM Bert encoder and Attention Decoder
class BertEncoder(nn.Module):
    def __init__(self, model_name, bert_tune_layers = 0, hidden_units = 512):
        super(BertEncoder, self).__init__()
        self.hidden_units = hidden_units
        self.bert = BertModel.from_pretrained(model_name)
        self.bert_tune_layers = bert_tune_layers
        self.bert_hidden_units = self.bert.config.hidden_size
        self.bert_layers = self.bert.config.num_hidden_layers

        if self.bert_tune_layers > 0:
            for i in range(self.bert_layers - self.bert_tune_layers):
                for param in self.bert.parameters():
                    param.requires_grad = False
                for param in self.encoder.layer[-self.bert_tune_layers:].parameters():
                    param.requires_grad = True

        self.fc = nn.Linear(self.bert_hidden_units, self.hidden_units)
        self.relu = nn.ReLU()

    def forward(self, x, attn_mask):
        with torch.no_grad():
            encoder_out = self.bert(x, attention_mask = attn_mask)
        encoder_out = encoder_out[0]
        encoder_out = self.fc(encoder_out)
        encoder_out = self.relu(encoder_out)
        return encoder_out
    
    
class AttentionNetworkBert(nn.Module):
    def __init__(self, hidden_units):
        super(AttentionNetworkBert, self).__init__()
        self.hidden_units = hidden_units
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.attn = nn.Linear(hidden_units * 2, hidden_units)
        self.v = nn.Linear(hidden_units, 1, bias=False)

    def forward(self, ht, encoder_out):
        ht = ht.repeat(encoder_out.shape[1], 1 , 1) # [seq_len, batch_size, hidden_units]

        ht = ht.transpose(0,1) # [batch_size, seq_len, hidden_units]

        energy = torch.cat([ht, encoder_out], dim=2)
        energy = self.attn(energy)
        energy = self.relu(energy)    
   
        attention = self.v(energy)
        attention = attention.squeeze(2)

        attention_weights = self.softmax(attention)
        attention_weights = attention_weights.unsqueeze(1)

        context = torch.bmm(attention_weights, encoder_out)

        return context, attention_weights
    

class AttenDecoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_units=1024, num_layers=1, p = 0.2, bidirectional=False):
        super(AttenDecoder, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
        self.attention = AttentionNetworkBert(hidden_units)
        self.LSTM = nn.LSTM(hidden_units + embed_dim, hidden_units, num_layers = num_layers, bidirectional = bidirectional, dropout=p, batch_first=True)
        self.fc = nn.Linear(hidden_units, input_size)

    def forward(self, x, h0_c0, encoder_out):

        x = x.unsqueeze(1)

        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        context, attention_weights = self.attention(h0_c0[0][-1].unsqueeze(0), encoder_out)

        lstm_input = torch.cat((embedded, context), dim = 2)

        output, h0_c0 = self.LSTM(lstm_input, h0_c0)
        output = self.fc(output)
        return output, h0_c0


#Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.embed_dim = args.embed_dim
        #encoder hidden is bert hidden dimenstion
        self.encoder_hidden_dim = args.en_hidden
        self.decoder_hidden_dim = args.de_hidden
        self.num_layers = args.en_num_layers
        self.dropout = 0.2
        self.processed_data = args.processed_data
        
    
        self.encoder_word2idx = self.get_encoder_word2idx()
        self.decoder_word2idx = self.get_decoder_word2idx()
        self.encoder_input_size = len(self.encoder_word2idx)
        self.decoder_input_size = len(self.decoder_word2idx)  

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

    def get_encoder_word2idx(self):
        with open(os.path.join(self.processed_data, "encoder_word2idx.pickle"), "rb") as file:
            word2idx = pickle.load(file)
        with open(os.path.join(self.processed_data, "encoder_idx2word.pickle"), "rb") as file:
            idx2word = pickle.load(file)
                
        
        return word2idx
    
    def get_decoder_word2idx(self):
        
        with open(os.path.join(self.processed_data, "decoder_word2idx.pickle"), "rb") as file:
            word2idx = pickle.load(file)
        with open(os.path.join(self.processed_data, "decoder_idx2word.pickle"), "rb") as file:
            idx2word = pickle.load(file)
        
        return word2idx
    
    def get_encoder(self):
        return BertEncoder("bert-base-uncased", bert_tune_layers = 3, hidden_units = self.encoder_hidden_dim)
    def get_decoder(self):
        return AttenDecoder(self.decoder_input_size, self.embed_dim, hidden_units = self.decoder_hidden_dim, num_layers = self.num_layers, p = self.dropout, bidirectional = False)
    
    
    def forward(self, src, trg, attn_mask, teacher_forcing_ratio = 0.6):
        batch_size = trg.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = len(self.decoder_word2idx)
        
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(device)

        encoder_out = self.encoder(src, attn_mask)
        hidden = torch.zeros(1, batch_size, self.decoder_hidden_dim).to(device)
        cell = torch.zeros(1, batch_size, self.decoder_hidden_dim).to(device)
        input = trg[:, 0]
        for t in range(1, max_len):
            output, (hidden, cell) = self.decoder(input, (hidden, cell), encoder_out)
            outputs[:, t, :] = output.squeeze(1)
            top1 = output.argmax(2)
            input = trg[:, t] if random.random() < teacher_forcing_ratio else top1.squeeze(1)
        return outputs
    

#TRAINING
def convert_idx_sentence(args, output, problem, linear_formula, answer, de_idx2word, en_idx2word, mode):
    """
    write to a file named my_output.json
    format for all problem should be
    {
        "Problem" : <problem>
        "answer" : <answer>
        "predicted" : <predicted_output>
        "linear_formula" : <original linear formula>
    }
    """
    output = output.cpu().detach().numpy()
    linear_formula = list(linear_formula)
    problem = list(problem)
    batch_size = output.shape[0]
    if(mode == "dev"):
        output_file = os.path.join("/home/cse/btech/cs1211010/a2/",f"dev_output.json")
    elif (mode == "test"):
        output_file = os.path.join("/home/cse/btech/cs1211010/a2/",f"test_output.json")
    else:
        print("Mode not recognized")
        return 

    # convert all the tensors into words usinf idx2word dictionary
    for b in range(batch_size):
        prb = ""
        ans = ""
        predicted = ""
        lf = ""
        for i in range(len(output[b])):
            temp = de_idx2word[(int)(output[b][i])]
            predicted += temp
            if(temp =="<eos>"):
                break

        for i in range(len(linear_formula[b])):
            temp = de_idx2word[(int)(linear_formula[b][i].item())]
            lf += temp 
            if(temp =="<eos>"):
                break

        for i in range(len(problem[b])):
            temp= en_idx2word[(int)(problem[b][i].item())]
            prb += temp + " "
            if(temp == "<eos>"):
                break

        
        ans += str(answer[b].item())

        data = {
            "Problem" : prb,
            "answer" : ans,
            "predicted" : predicted,
            "linear_formula" : lf
        }

        with open(output_file, "a") as file:
            json.dump(data, file)
            file.write("\n")
    return

def convert_idx_sentence_infer(args, output, problem, de_idx2word, en_idx2word):
    """
    write to a file named my_output.json
    format for all problem should be
    {
        "Problem" : <problem>
        "answer" : <answer>
        "predicted" : <predicted_output>
        "linear_formula" : <original linear formula>
    }
    """
    output = output.cpu().detach().numpy()
    linear_formula = list(linear_formula)
    problem = list(problem)
    batch_size = output.shape[0]
    predicte_output_list =[]

    # convert all the tensors into words usinf idx2word dictionary
    for b in range(batch_size):
        prb = ""
        ans = ""
        predicted = ""
        lf = ""
        for i in range(len(output[b])):
            temp = de_idx2word[(int)(output[b][i])]
            predicted += temp
            if(temp =="<eos>"):
                break

        for i in range(len(problem[b])):
            temp= en_idx2word[(int)(problem[b][i].item())]
            prb += temp + " "
            if(temp == "<eos>"):
                break

        #write problema and predicted output to the list
        predicte_output_list.append((prb, predicted))

    return predicte_output_list

def beam_search(args, model,encoder_out, en_ht, en_ct, start_token, end_token, max_target_len = 500, beam_size = 10):
    beam = [([start_token], (en_ht, en_ct), 0)]
    # print("Beam Search input hidden and cell shape, start and end tokens", en_ht.shape, en_ct.shape, start_token, end_token)

    i = 0
    while i < max_target_len -1:
        new_beam = []
        for sequence, (ht, ct), score in beam:
            prev_token = [sequence[-1]] #get first token for each beam
            prev_token = torch.LongTensor(prev_token).to(device)

            decoder_out, ht, ct = model.decoder(prev_token, (ht, ct), encoder_out) #pass through decoder

            decoder_out = decoder_out.squeeze(1)
            top_info = decoder_out.topk(beam_size, dim=1) #get top k=beam_size possible word indices and their values
            top_vals, top_inds = top_info

            for j in range(beam_size):
                new_word_idx = top_inds[0][j]                
                new_seq = sequence + [new_word_idx.item()]
                new_word_prob = torch.log(top_vals[0][j])
                updated_score = score - new_word_prob
                new_candidate = (new_seq, (ht, ct), updated_score)
                new_beam.append(new_candidate)

        # new_beam = sorted(new_beam, reverse=True, key=lambda x: x[2])
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_size]
        i += 1

    best_candidate = beam[0][0] #return best candidate based on score
    decoded_words = torch.zeros(1, max_target_len)

    for t in range(max_target_len):
        decoded_words[:, t] = torch.LongTensor([best_candidate[t]])
    
    return decoded_words

def evaluator_dev(args, model):
    # on current model given , need to modify for any model
    dev_dataset = TextToMathDataset(args.processed_data, "dev")
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    de_word2idx = dev_dataset.de_word2idx
    de_idx2word = dev_dataset.de_idx2word
    en_idx2word = dev_dataset.en_idx2word

    model.eval()

    print("Evaluating model on val data on given model")

    start_token = de_word2idx["<sos>"]
    end_token = de_word2idx["<eos>"]

    target_vocab_size = len(de_word2idx)
    decoder_hidden_units = args.de_hidden

    #file to be made
    my_file = "my_output.json"
    my_file = os.path.join(args.processed_data, my_file)

    for i, batch in enumerate(dev_loader):
        problem = batch['problem'].to(device)
        linear_formula = batch['linear_formula'].to(device)
        prb_attn_mask = batch['problem_attn_mask'].to(device)
        answer = batch['answer'].to(device)
        
        batch_size = problem.shape[0]
        max_target_len = linear_formula.shape[1]
#         max_target_len = 500

        words = torch.zeros(batch_size, max_target_len).to(device)

        output  = model.encoder(problem, prb_attn_mask) 
        hidden = torch.zeros(1, batch_size, decoder_hidden_units).to(device) 
        cell = torch.zeros(1, batch_size, decoder_hidden_units).to(device)
        #beam search
        for b in range(batch_size):
            print(f"at i: {i} and inside batch :{b}")
            words[b,:] = beam_search(args, model ,output[b,:,:].unsqueeze(0), hidden[:,b,:].unsqueeze(1),cell[:,b,:].unsqueeze(1), start_token, end_token, max_target_len = max_target_len, beam_size = 10)
        # convert_idx_sentence(args, words, problem, linear_formula, answer)
        convert_idx_sentence(args, words, problem, linear_formula, answer, de_idx2word, en_idx2word, "dev")
    return


#train
def train_S2S():
    print(device)
    args = ARGS()
    pth = "/home/cse/btech/cs1211010/a2/math-data"
    args.processed_data = os.path.join(pth, "processed_data")
    # args.data_dir = os.path.join(pth, "data")
    print(args.processed_data)
    train_dataset = TextToMathDataset(args.processed_data, "train")
    dev_dataset = TextToMathDataset(args.processed_data, "dev")
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, collate_fn=collate)

    #------------------------------------------------
    model = Seq2Seq(args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) 
    schedulers = {
        "stepLR" : torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=- 1, verbose=False),
        "cosineLR" : torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, verbose=False)
    }

    current_scheduler = schedulers["cosineLR"]
    # ----------------------------------
    loss_tracker = defaultdict(list)
    # val_accuracy_tracker = defaultdict(list)
    min_loss = 1000000
    best_epoch = 0
    for epoch in range(args.epochs):
        print("\n\n-------------------Epoch = ", epoch, "------------------------------\n")
        model.train()
        epoch_loss =[]
        total_loss = 0
        for i, batch in enumerate(train_loader):
#             if(i==4) :  break 
            optimizer.zero_grad()

            prb = batch["problem"].to(device)
            lf = batch["linear_formula"].to(device)
            prb_attn_mask = batch["problem_attn_mask"].to(device)

            output  = model(prb, lf, prb_attn_mask)

            output = output.reshape(-1, output.shape[2])
            lf = lf.reshape(-1)
            
            loss = criterion(output, lf)
            loss.backward()
            epoch_loss.append(loss.item())
            total_loss += loss.item()
            optimizer.step()

        current_scheduler.step()
        print(f"Epoch {epoch}, Loss = {total_loss/len(train_loader)}")

        dev_loss =[]
        model.eval()
        for i, batch in enumerate(dev_loader):
            prb = batch["problem"].to(device)
            lf = batch["linear_formula"].to(device)
            prb_attn_mask = batch["problem_attn_mask"].to(device)

            output  = model(prb, lf, prb_attn_mask)

            output = output.reshape(-1, output.shape[2])
            lf = lf.reshape(-1)

            loss = criterion(output, lf)
            dev_loss.append(loss.item())



        avg_dev_loss = np.mean(dev_loss)
        avg_epoch_loss = np.mean(epoch_loss)

        loss_tracker["train"].append(avg_epoch_loss)
        loss_tracker["dev"].append(avg_dev_loss)

        print(f"Epoch {epoch}, Train Loss = {avg_epoch_loss}, Dev Loss = {avg_dev_loss}")
        # print(f"Epoch Time = {end2 - end1}")
        # loss_tracker["dev"].append(avg_dev_loss)

        with open(os.path.join(args.result_dir, "loss_tracker{}.json".format(args.model_type)), "w") as outfile:
            json.dump(loss_tracker, outfile)

        torch.save(model, os.path.join(args.checkpoint_dir, "latest_checkpoint_{}.pth".format(args.model_type)))

        model_state = {
                'epoch': epoch,
                'train_loss' : avg_epoch_loss,
                'prev_best_epoch': best_epoch
        }

        with open(os.path.join(args.checkpoint_dir, "latest_chkpt_status_{}.json".format(args.model_type)), "w") as outfile:
            json.dump(model_state, outfile)

        #save the model whose loss is minimum
        if avg_dev_loss < min_loss:
            min_loss = avg_dev_loss
            best_epoch = epoch
            torch.save(model, os.path.join(args.checkpoint_dir, "best_checkpoint_{}.pth".format(args.model_type)))
            print("Best Model saved at epoch = ", epoch)

    print("Training and Dev Complete for Seq2Seq Bert model")
    
    #run evaluation on dev data using best model
    model = torch.load(os.path.join(args.checkpoint_dir, "best_checkpoint_{}.pth".format(args.model_type))).to(device)
    evaluator_dev(args, model)

    #run on latest model
    # model = torch.load(os.path.join(args.checkpoint_dir, "latest_checkpoint_{}.pth".format(args.model_type))).to(device)
    # evaluator_dev(args, model)
    return args, model
    
#test
def test_S2S(test_data_file,model):
    # print(device)
    args = ARGS()
    pth = os.getcwd()
    args.processed_data = os.path.join(pth, "processed_data")
#     args.data_dir = os.path.join(pth, "data")
    print(args.processed_data)

    file_data = args.processed_data

    test_to_load = test_data_file

    test_dataset = TextToMathDataset(file_data, test_to_load)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, collate_fn=collate)
    de_word2idx = test_dataset.de_word2idx
    en_word2idx = test_dataset.en_word2idx
    
    de_idx2word = test_dataset.de_idx2word
    en_idx2word = test_dataset.en_idx2word
    # decoder_hidden_units = model.decoder_hidden_units
    #------------------------------------------------
    # model = torch.load(os.path.join(args.checkpoint_dir, "latest_checkpoint_Seq2Seq.pth")).to(device) #load the latest checkpoint
    # model = torch.load(os.path.join(args.checkpoint_dir, "best_checkpoint_Seq2Seq.pth")).to(device) #load the best checkpoint
    # my_file = ".json"
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_tracker = defaultdict(list)
    final_predicte_output_list = [] 

    start_token = de_word2idx["<sos>"]
    end_token = de_word2idx["<eos>"]
    print(f" Start Token {start_token} and end token {end_token}")
    for i, batch in enumerate(test_loader):
        if(i==3) : break
        problem = batch['problem'].to(device)
        # linear_formula = batch['linear_formula'].to(device)
        prb_attn_mask = batch['problem_attn_mask'].to(device)
        # answer = batch['answer'].to(device)

        batch_size = problem.shape[0]
        # max_target_len = linear_formula.shape[1]
        max_target_len = 500

        words = torch.zeros(batch_size, max_target_len).to(device)
        output ,hidden, cell = model.encoder(problem, prb_attn_mask)
        hidden = torch.zeros(1, batch_size, model.decoder_hidden_dim).to(device)
        cell = torch.zeros(1, batch_size, model.decoder_hidden_dim).to(device)

        #beam search
        for b in range(batch_size):
            print(f" i : {i} , b : {b}")
            words[b,:] = beam_search(args, model ,output[b,:,:].unsqueeze(0), hidden[:,b,:].unsqueeze(1),cell[:,b,:].unsqueeze(1), start_token, end_token, max_target_len = max_target_len, beam_size = 10)
        predicte_output_list = convert_idx_sentence_infer(args, words, problem, de_idx2word, en_idx2word)
        final_predicte_output_list.extend(predicte_output_list)
    # print("Running evaluation script for test ... ")
    # subprocess.call(f"python3 evaluator.py {my_file}")
    #Write the final_predicte_output_list to the test to load file in json format
    with open(test_to_load, "w") as file:
        
        for item in final_predicte_output_list:
            json.dump(item, file)
            file.write("\n")
    print("Testing Complete. JSON created")
    return

# if __name__ == "__main__":
#     args,model = train_S2S()
#     test_S2S(args,model)


def run_test_BertFrozen(model_path , test_data_file):
    cwd  = os.getcwd()
    model_path = os.path.join(cwd,model_path) 
    model = torch.load(model_path)
    test_S2S(test_data_file,model)
    return


def run_test_BertTuned(model_path , test_data_file):
    cwd  = os.getcwd()
    model_path = os.path.join(cwd,model_path) 
    model = torch.load(model_path)
    test_S2S(test_data_file,model)
    return