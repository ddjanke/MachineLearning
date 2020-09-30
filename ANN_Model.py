import numpy as np
import matplotlib.pyplot as plt
import gc

import torch
from torch.nn import functional as F
import ergence_feature_extraction as erg

def count_synapses(nodes):
    tot = 0
    for i,n in enumerate(nodes[:-1]):
        tot += (n+1)*nodes[i+1]
    return tot

def compressed_relu(x, gain = 1):
    return F.relu(x)*gain

class CompressedTanh(torch.nn.Tanh):
    def __init__(self, offset = 0, max_range = 2, compression = 1, inplace = False):
        super(CompressedTanh, self).__init__()
        self.max_range = max_range
        self.offset = offset
        self.compression = compression
    
    def forward(self,x):
        return (self.max_range/2)*torch.tanh(x*self.compression) + self.offset

class ANNModelTorch(torch.nn.Module):
    def __init__(self, nodes, device = 'cuda', activation = 'Tanh', dropout = False):
        super(ANNModelTorch,self).__init__()
        self.nodes = nodes
        self.synapses = count_synapses(nodes)
        if activation == 'Tanh':
            self.activation = CompressedTanh() # CompressedTanh(0,0.2,1)
            self.threshold = 0.5
        elif activation == "CompTanh":
            self.activation = CompressedTanh(0, 1, 0.5) # 
            self.threshold = 0.3
        else:
            self.activation = CompressedTanh(0, 0.5, 0.25) # 
            self.threshold = 0.05
        self.drop_rate = dropout
        self.device = device
        self.initialize_model()
        
    def initialize_model(self):
        layers = []
        for i,ns in enumerate(self.nodes[1:]):
            layers += [Linear_wVoltages(self.nodes[i], ns, layer = i,
                                        bias = True, device = self.device),
                       self.activation,
                       torch.nn.Dropout(p = self.drop_rate)]
        #layers[-1] = CompressedTanh(0.5, 1, 1)
        layers = layers[:-1] #+ [torch.nn.Sigmoid()]
        
        self.model = torch.nn.Sequential(*layers).to(self.device)
        self.model.apply(self.initialize_weights)
        
    def initialize_weights(self, m):
        if 'weight' in m.state_dict():
            torch.nn.init.xavier_uniform_(m.weight, gain = 1)
            m.bias.data.fill_(0.001)
        
    def forward(self,X,device = None):
        if device is None: device = self.device
        if type(X) is np.ndarray:
            X = torch.as_tensor(X, dtype = torch.float32).to(device)
        return self.model.forward(X).squeeze()
    
    
    def fit(self, generator, epochs, batch_size = 1, verbose = False,
            generator_kwargs = {"mode" : "train"},
            weight_decay = 0, learning_rate = 0.001,
            early_stop_train = False, early_stop_eval = False):
        
        early_stop = early_stop_train or early_stop_eval
        self.model = self.model.to(self.device)
        
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr = 0.002,
                                        weight_decay = weight_decay)
        if early_stop_eval:
            generator_kwargs["mode"] = "eval"
        
        self.model.train()
        es = myEarlyStop(patience = 100, delta = 0.0001)
        #print(self.model.state_dict())
        for epoch in range(epochs):
            acc = 0
            for g in generator(**generator_kwargs):
                if early_stop_eval: i, Xi, yi, Xv, yv = g
                else: i, Xi, yi = g
                pos_weight = erg.AudioFeatureGenerator.calc_pos_weight(yi)
                if yi.max() == 1:
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight,
                                                   dtype = torch.float32))
                else:
                    criterion = torch.nn.CrossEntropyLoss()
                Xi = torch.as_tensor(Xi, dtype = torch.float32).to(self.device)
                yi = torch.as_tensor(yi, dtype = torch.long).to(self.device)
                y_pred = self.forward(Xi)
                loss = criterion(y_pred, yi)
                loss = loss / batch_size
                loss.backward()
                if early_stop_train:
                    acc += self.accuracy(yi, y_pred).item()
                elif early_stop_eval:
                    Xv = torch.as_tensor(Xv, dtype = torch.float32).to(self.device)
                    yv = torch.as_tensor(yv, dtype = torch.float32).to(self.device)
                    yv_pred = self.forward(Xv)
                    acc += self.accuracy(yv, yv_pred)
                    del Xv; del yv; del yv_pred
                
                
                del Xi
                del yi
                del y_pred
                gc.collect()
                if (i+1) % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            if early_stop:
                es(acc, self.model)
                #print(str(acc)[:7], end = ' ')
            if es.stop or ((epoch + 1) == epochs and early_stop):
                self.model.load_state_dict(es.es_dict)
                print(" Exiting training: early stop triggered.")
                break
        

            
            if verbose > 0:
                if epoch % verbose == 0:
                    print('Epoch {} train loss: {}'.format(epoch, loss.item()))
                    
                    
    def population_fit(self, generator, epochs, population = 1, verbose = False,
                        generator_kwargs = {"mode" : "train"},
                        weight_decay = 0, learning_rate = 0.001,
                        early_stop_train = False):
        
        self.model = self.model.to(self.device)
        
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr = 0.002,
                                        weight_decay = weight_decay)
        
        self.model.train()
        es = myEarlyStop(patience = 100, delta = 0.0001, mode = "loss")
        #print(self.model.state_dict())
        for epoch in range(epochs):
            total_loss = 0
            for p in range(population):
                self.add_noise()
                for g in generator(**generator_kwargs):
                    i, Xi, yi = g
                    pos_weight = erg.AudioFeatureGenerator.calc_pos_weight(yi)
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight,
                                               dtype = torch.float32))
                    Xi = torch.as_tensor(Xi, dtype = torch.float32).to(self.device)
                    yi = torch.as_tensor(yi, dtype = torch.float32).to(self.device)
                    y_pred = self.forward(Xi)
                    loss = criterion(y_pred, yi)
                    loss = loss / population
                    total_loss += loss.item()
                    loss.backward()
                    
                    
                    del Xi
                    del yi
                    del y_pred
                    gc.collect()
                
            if early_stop_train:
                es(total_loss, self.model)
                
            optimizer.step()
            optimizer.zero_grad()
            
            if es.stop or ((epoch + 1) == epochs and early_stop_train):
                self.model.load_state_dict(es.es_dict)
                print(" Exiting training: early stop triggered.")
                break
            
            if verbose > 0:
                if epoch % verbose == 0:
                    print('Epoch {} train loss: {}'.format(epoch, total_loss))          

                    
    
    def predict_accuracy(self, generator, generator_kwargs = {"mode" : "eval"},
                         feature_noise = 1):
        self.model.eval()
        acc = torch.as_tensor([0, 0], dtype = torch.float32)
        for g in generator(**generator_kwargs):
            if (len(g) == 3):
                i, X, y = g
                
            if (len(g) == 5):
                i, X, y, Xv, yv = g
                
            y = torch.as_tensor(y, dtype = torch.float32).to(self.device)
            y_pred = self.predict(X * feature_noise)
            
            acc[0] = acc[0] + self.accuracy(y, y_pred)
            
            if (len(g) == 5):
                yv = torch.as_tensor(yv, dtype = torch.float32).to(self.device)
                yv_pred = self.predict(Xv * feature_noise)
                
                acc[1] = acc[1] + self.accuracy(yv, yv_pred)
                
        if (len(g) == 3):
            #print(torch.mean(y_pred).data, end = " ")
            return acc[0].item() / (i + 1)
        if (len(g) == 5):
            #print(torch.mean(y_pred).cpu().numpy(), end = " ")
            #print(torch.mean(yv_pred).cpu().numpy(), end = " ")
            return acc.data / (i + 1)
    
    
    def accuracy(self, y, y_pred):
        if len(y_pred.shape) == 1:
            y_pred = (y_pred > self.threshold).float()
        else:
            y_pred = torch.argmax(y_pred, dim = 1)
        
        return torch.sum(torch.eq(y_pred, y).float()) / len(y)
    
    
    def population_accuracy(self, generator, generator_kwargs, population = 1,
                            feature_noise = None, feature_noise_params = [1, 0.1],
                            weight_noise = None, average = True):
        if feature_noise is None: feature_noise = np.array([1])
        if type(feature_noise) is bool:
            if feature_noise == True: feature_noise = np.array([1,1])
        
        self.model.eval()
        
        if average:
            acc = torch.as_tensor(0, dtype = torch.float32)
        else:
            acc = [0] * population
        
        dset = generator_kwargs.get("data_set", "train")
        print("Testing {}...".format(dset), end = " ")
        for p in range(population):
            if weight_noise is not None:
                self.add_noise(wnoise = weight_noise[p])
            else:
                self.add_noise()
            for g in generator(**generator_kwargs):
                i, X, y = g
                    
                if feature_noise.shape[0] != X.shape[1]:
                    feature_noise = np.random.normal(loc = feature_noise_params[0],
                                                     scale = feature_noise_params[1],
                                                     size = X.shape[1])
                    
                y_pred = self.predict(X * feature_noise)
                y = torch.as_tensor(y, dtype = torch.float32).to(self.device)
                if average:
                    acc = acc + self.accuracy(y, y_pred)
                else:
                    acc[p] = self.accuracy(y, y_pred).item()
            if (p + 1) % 100 == 0: print(p + 1, end = ' ')
        print()
        if average:    
            return acc.item() / (i + 1) / population
        else:
            return acc

    
    def predict_accuracy_all(self,X,y):
        self.model.eval()
        self.model.cpu()
        accs = []
        for Xi,yi in zip(X,y):
            if type(yi) is np.ndarray:
                yi = torch.as_tensor(yi, dtype = torch.float32)
            self.model.eval()
            y_pred = (self.forward(Xi,'cpu') > self.threshold).float()
            acci = torch.sum(torch.eq(y_pred,yi).float())/len(yi)
            accs += [acci.data]
        return np.array(accs)
        
    def predict(self, X):
        self.model.eval()
        y_pred = (self.forward(X) > self.threshold).float()
        
        return y_pred
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def add_noise(self, wnoise = None, how = "both", loc_scale = [1, 0.035, 0, 1 * 0.07]):
        if wnoise is not None:
            size = 2 if how == "both" else 1
            if self.synapses * size != len(wnoise):
                raise TypeError('Length of noise arrays ({}) does not match number of synapses ({}) in module.'\
                                .format(len(wnoise), self.synapses))
        elif how == "both":
            wnoise = np.random.normal(loc = loc_scale[0],
                                      scale = loc_scale[1],
                                      size = self.synapses)
            
            wnoise = np.concatenate([wnoise,
                                     np.random.normal(loc = loc_scale[2],
                                                      scale = loc_scale[3],
                                                      size = self.synapses)])
        else:
            wnoise = np.random.normal(loc_scale[0],
                                      scale = loc_scale[1],
                                      size = self.synapses)
            
        start = 0
        for module in self.model:
            if 'weight' in module.state_dict():
                synapses = module.bias.shape[0]+module.weight.shape[0]*module.weight.shape[1]
                all_noise = wnoise[start : start + synapses] if how != "both" else \
                    np.concatenate([wnoise[start : start + synapses],
                                    wnoise[start + self.synapses : start \
                                           + self.synapses + synapses]])
                        
                module.add_noise(all_noise, noise_mode = how)
                start += synapses
    
class Linear_wVoltages(torch.nn.Linear):
    def __init__(self,in_features, out_features, layer, bias=True, device = 'cuda'):
        super(Linear_wVoltages,self).__init__(in_features, out_features, bias)
        self.device = device
        #self.activation = torch.nn.Tanh()
        self.layer = layer
        self.shape = (out_features,in_features + 1)
        self.initialize()
        
    def initialize(self):
        self.maxgain = 10
        self.rate = 1 / self.maxgain
        
        #self.all_rates = self.rate*torch.ones(self.shape).to(self.device)
        self.all_gains = self.maxgain*torch.ones(self.shape).to(self.device)
        self.all_add = torch.zeros(self.shape).to(self.device)
        
            
    def forward(self,X):
        weight = self.all_gains[:,1:].to(X.device)*torch.tanh(self.rate*self.weight) + self.all_add[:,1:]
        bias = self.all_gains[:,0].to(X.device)*torch.tanh(self.rate*self.bias) + self.all_add[:,0]
        return F.linear(X, weight, bias)
    
    def add_noise(self, gnoise, noise_mode = 'both'):
        if type(gnoise) is np.ndarray:
            gnoise = torch.as_tensor(gnoise, dtype = torch.float32)
        
        if noise_mode == "both":
            mult = gnoise[:self.shape[0] * self.shape[1]]
            add = gnoise[self.shape[0] * self.shape[1]:]
            self.all_gains = (self.maxgain * mult.view(self.shape)).to(self.device)
            self.all_add =  add.view(self.shape).to(self.device)
        
        if noise_mode == 'add':
            gnoise = (gnoise - 1) * self.maxgain
            self.all_gains = (self.maxgain*torch.ones(self.shape)+gnoise.view(self.shape)).to(self.device)
            
        if noise_mode == 'mult':
            #self.all_rates = (self.rate*torch.ones(self.shape)*rnoise.view(self.shape)).to(self.device)
            self.all_gains = (self.maxgain*torch.ones(self.shape)*gnoise.view(self.shape)).to(self.device)

class myEarlyStop():
    def __init__(self, patience = 50, delta = 0.0001, mode = "acc"):
        self.patience = patience
        self.delta = delta
        self.es_counter = 0
        self.es_dict = {}
        if mode == "acc":
            self.best_score = 0
        if mode == "loss":
            self.best_score = -np.inf
        self.mode = mode
        self.stop = False
        
    def __call__(self, score, model):
        if self.mode == "loss": score *= -1
        if score > self.best_score + self.delta:
            self.es_counter = 0
            self.best_score = score
            self.es_dict = model.state_dict()
            
        else:
            self.es_counter += 1
            
        if self.es_counter > self.patience:
            self.stop = True