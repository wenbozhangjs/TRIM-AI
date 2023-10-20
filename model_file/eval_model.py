import os,random,time,argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AdamW
from transformers import XLMRobertaTokenizer,XLMRobertaConfig,XLMRobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")# remove UndefinedMetricWarning: Precision is ill-defined

class ProcessData:
	'''
	class for dataset processing or unseen new data processing
	'''
	def __init__(self,tokenizer_vocab_path,tokenizer_max_len,feature_name):
		self.tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_vocab_path) # load the pretained model
		self.max_len = tokenizer_max_len
		self.feature_name = feature_name

	def load_train_data(self,path):
		'''
		Load the dataset for training and return list of original text and corresponding label
		'''
		df = pd.read_csv(path)
		df=df.dropna(subset=['Translation'])
		text_list = df[self.feature_name].to_list()
		type_label = df['Intent1_Cleaned'].value_counts().index.to_list()
		label_encoder = preprocessing.LabelEncoder()
		label_encoder = label_encoder.fit(type_label)
		label_tmp = label_encoder.transform(df['Intent1_Cleaned'])
		labels = pd.Series(label_tmp).to_list()
		
		# add label-intent transfer table
		df1 = pd.DataFrame(type_label,columns=['intent'])
		df2 = pd.DataFrame(label_encoder.transform(type_label),columns=['corresponding_label'])
		df = pd.concat([df1,df2],axis=1)
		df.to_csv('intent_label_relation.csv', index=False)
		return text_list, labels

	def load_test_data(self,path):
		'''
		Load the data for testing or new data for prediction
		'''
		df = pd.read_csv(path)
		text_list = df['text'].to_list()
		label_list = df['label'].to_list()
		return text_list, label_list
	
	def encode_fn(self,text_list):
		'''
		Using tokenizer to preprocess the text 
		example of text_list:['today is a good daya','I'm at psu']
		'''
		tokenizer = self.tokenizer(text_list,
									padding = True,
									truncation = True,
									max_length = self.max_len,
									return_tensors='pt'
									)
		input_ids = tokenizer['input_ids']
		attention_mask = tokenizer['attention_mask']
		return input_ids,attention_mask
  
	def process_train_data(self, text_list, labels):
		'''
		Preprocess text and prepare dataloader for training set
		'''
		input_ids,attention_mask = self.encode_fn(text_list)
		labels = torch.tensor(labels)
		data = TensorDataset(input_ids,attention_mask,labels)
		return data
	
	def process_test_data(self, text_list, labels):
		'''
		Preprocess text and prepare dataloader for testing set or unseen new data
		'''
		input_ids,attention_mask = self.encode_fn(text_list)
		data = TensorDataset(input_ids,attention_mask)
		return data,labels

	def split_dataset(self, data, labels):
		'''
		Split the dataset into training set and validation set
		'''
		train_x, val_x, train_y, val_y = train_test_split(data,labels,test_size = 0.2,random_state = 0)
		return train_x, val_x, train_y, val_y
	
	def save_splited_data(self,data,label,path,file_name):
		'''
		Save the training data and test data into csv
		'''
		df1=pd.DataFrame(data,columns=['text'])
		df2=pd.DataFrame(label,columns=['label'])
		df=pd.concat([df1,df2],axis=1)
		path_save= os.path.join(path,file_name)
		df.to_csv(path_save, index=False)

	def prepare_data_loader(self, path_data, path_save_split,mode_test=False):
		'''
		Prepare dataloader for training or test
		'''
		if mode_test:
			# load dataset
			text_list, labels = self.load_test_data(path_data)
			
			# preprocessing and prepare dataloader
			test, labels = self.process_test_data(text_list, labels)
			return test, labels
		else:
			# load dataset
			text_list, labels = self.load_train_data(path_data)
			
			# preprocessing and prepare dataloader
			train_x, val_x, train_y, val_y = self.split_dataset(text_list, labels)
			
			# if save the new splitted data into training data and test data
			if path_save_split:

				self.save_splited_data(train_x,train_y,path_save_split,'split_train_data.csv')
				self.save_splited_data(val_x,val_y,path_save_split,'split_test_data.csv')
				
			train = self.process_train_data(train_x, train_y)
			validation = self.process_train_data(val_x, val_y)
			return train,validation

class My_XLMRobertaModel:
	def __init__(self,pretrain_model_name,
	                  path_saved_model,
					  learning_rate,
					  n_class,
					  epochs,
					  warm_step,
					  train_batch_size,
					  test_batch_size,
					  max_len,
					  gpu=True):
		self.n_class = n_class # number of class for classification
		self.max_len = max_len # max sentence length
		self.lr = learning_rate # learning rate
		self.epochs = epochs
		self.warm_step=warm_step
		
		self.tokenizer = XLMRobertaTokenizer.from_pretrained(pretrain_model_name) # load the pretained model
		self.batch_size = train_batch_size # batch_size for model training
		self.val_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
			
		self.path_saved_model = path_saved_model # path to save model checkpoints
		self.gpu = gpu # if using GPU for model training
		config = XLMRobertaConfig.from_pretrained(pretrain_model_name) # obtain Roberta model config
		config.num_labels = n_class
		self.model = XLMRobertaForSequenceClassification.from_pretrained(pretrain_model_name,config=config)

		if self.gpu:
			seed = 42
			random.seed(seed)
			np.random.seed(seed)
			torch.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True
			self.device = torch.device('cuda')
		else:
			self.device = 'cpu'

	def flat_accuracy(self, preds, labels):
		'''
		Evaluate model performance based on accuracy scores
		'''
		pred_flat = np.argmax(preds, axis=1).flatten()
		labels_flat = labels.flatten()
		return accuracy_score(labels_flat, pred_flat)

	def flat_accuracy_new(self, preds, labels):
		'''
		Evaluate model performance based on accracy, weighted average precsion, recall and F1 score
		'''
		pred_flat = np.argmax(preds, axis=1).flatten()
		labels_flat = labels.flatten()
		acc = accuracy_score(labels_flat, pred_flat)
		pre = precision_score(labels_flat, pred_flat,average='weighted')
		rec = recall_score(labels_flat, pred_flat, average='weighted')
		f1 = f1_score(labels_flat, pred_flat, average='weighted')
		return 	acc,pre,rec,f1

	def train_model(self,train,validation):
		'''
		Fine-tune the pretrained model based on new training data
		'''
		if self.gpu:
			self.model.cuda()
		optimizer = AdamW(self.model.parameters(), lr=self.lr)
		trainData = DataLoader(train, batch_size = self.batch_size, shuffle = True)
		valData = DataLoader(validation, batch_size = self.val_batch_size, shuffle = True)

		total_steps = len(trainData) * self.epochs
		scheduler = get_linear_schedule_with_warmup(optimizer, 
													num_warmup_steps=self.warm_step, 
													num_training_steps=total_steps)

		for epoch in range(self.epochs):
			total_eval_accuracy = 0
			total_eval_precision = 0
			total_eval_recall = 0
			total_eval_f1 = 0
			list_logits = []
			list_label_ids = []

			# start training the model
			self.model.train()
			total_loss, total_val_loss = 0, 0
			
			print('.... Start training at epoch-%-3d ....'%epoch)
			total_step_number=len(trainData)
			print('The epoch %-3d has total steps of: %-5d' %  (epoch ,total_step_number))
			
			for step,batch in enumerate(trainData):
				optimizer.zero_grad()
				outputs  = self.model(input_ids = batch[0].to(self.device),
									  attention_mask=batch[1].to(self.device),
									  labels=batch[2].to(self.device)
									  )   
				loss, logits = outputs.loss, outputs.logits
				total_loss += loss.item()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
				optimizer.step()
				scheduler.step()
				if step % 10 == 0 and step > 0: 
					self.model.eval()
					logits = logits.detach().cpu().numpy()
					label_ids = batch[2].cuda().data.cpu().numpy()
					avg_val_accuracy = self.flat_accuracy(logits, label_ids)
					
					print('Step: %-5d of total step:%-5d in epoch-%-3d' % (step,total_step_number,epoch))
					print(f'Accuracy: {avg_val_accuracy:.4f}')
					print('\n')

		    #Evaluate model on validation set after each training epoch
			self.model.eval()
			print('.... Testing ....')
			for i, batch in enumerate(valData):
				with torch.no_grad():
					outputs = self.model(input_ids=batch[0].to(self.device),
										 attention_mask=batch[1].to(self.device),
										 labels=batch[2].to(self.device)
										)
					loss, logits = outputs.loss, outputs.logits
					total_val_loss += loss.item()

					logits = logits.detach().cpu().numpy()
					label_ids = batch[2].cuda().data.cpu().numpy()

					acc,pre,rec,f1 = self.flat_accuracy_new(logits, label_ids)
					total_eval_accuracy += acc
					total_eval_precision += pre
					total_eval_recall += rec
					total_eval_f1 += f1
					list_logits += list(logits)#type_2
					list_label_ids += list(label_ids)#type_2

			avg_train_loss = total_loss / len(trainData)
			avg_val_loss = total_val_loss / len(valData)
			avg_val_accuracy = total_eval_accuracy / len(valData)
			avg_val_precision = total_eval_precision / len(valData)
			avg_val_recall = total_eval_recall / len(valData)
			avg_val_f1 = total_eval_f1 / len(valData)
			acc,pre,rec,f1 = self.flat_accuracy_new(np.array(list_logits), np.array(list_label_ids))#type_2

			print('.... Model summary on validation set ....')
			print(f'Train loss     : {avg_train_loss}')
			print(f'Validation loss: {avg_val_loss}')
			print(f'Accuracy:%.4f; Precision:%.4f; Recall:%.4f; F1_score:%.4f;' % (avg_val_accuracy,
			                                                                       avg_val_precision,
																				   avg_val_recall,
																				   avg_val_f1))
			print(f'[Whole set]Accuracy:%.4f; Precision:%.4f; Recall:%.4f; F1_score:%.4f;' % (acc,pre,rec,f1))
			print('\n')
			self.save_model(self.path_saved_model + '-' + str(epoch))

	def save_model(self , path):
		'''
		Save fine-tuned pretained model
		'''
		self.model.save_pretrained(path)
		self.tokenizer.save_pretrained(path)

	def load_model(self,path):
		'''
		Load tokenizer and pretrained BERT model
		'''
		model = XLMRobertaForSequenceClassification.from_pretrained(path)
		return model

	def eval_model(self,model,pred_data,y_true):
		'''
		Evaluate model based on precison, recall, and F1 score
		'''
		preds = self.predict_batch(model, pred_data)
		print(classification_report(y_true,preds))
		return preds

	def predict_batch(self, model, pred_data):
		''' 
		Model inference for new dataset
		'''
		pred_dataloader = DataLoader(pred_data, batch_size=self.test_batch_size, shuffle=False)
		model = model.to(self.device)
		model.eval()
		preds = []
		print('.... Inference on new unseen data ....')
		for i, batch in enumerate(tqdm(pred_dataloader)):
			with torch.no_grad():
				outputs = model(input_ids=batch[0].to(self.device),
								attention_mask=batch[1].to(self.device)
								)
				loss, logits = outputs.loss, outputs.logits
				logits = logits.detach().cpu().numpy()
				preds += list(np.argmax(logits, axis=1))
		return preds

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path_project_folder', default=r"/home/TRIM-AI/")
	parser.add_argument('--name_dataset', default="split_test_data.csv") # during training, a test set will be automatically splitted and saved in project folder
	parser.add_argument('--pretrained_model_name', default='xlm-roberta-large')
	parser.add_argument('--best_model_folder', default='xlm-roberta-large-2')
	parser.add_argument('--data_feature_type', default='Original Text') # could be'Translation'or 'Original Text'
	parser.add_argument('--data_prepare_mode', default=True) # False means prepare data for training 
	parser.add_argument('--num_train_epoch', type=int, default=3)
	parser.add_argument('--num_warm_step', type=int, default=100)
	parser.add_argument('--train_lr', type=int, default=2e-5)
	parser.add_argument('--num_class', type=int, default=58) # number of classes for text classification
	parser.add_argument('--batch_size_tra', type=int, default=8) # training and validation batch size
	parser.add_argument('--batch_size_tes', type=int, default=128) # inference batch size
	parser.add_argument('--token_max_len', type=int, default=250) # max_len that tokenizer will process
	parser.add_argument('--if_gpu', default=True) # max_len that tokenizer will process
	args = parser.parse_args()
	start_time = time.time()

    # set the dataset path
	path_test_data = os.path.join(args.path_project_folder, args.name_dataset)

	# set trained model path
	save_model_path = os.path.join(args.path_project_folder, "checkpoint")
	path_saved = os.path.join(save_model_path, args.pretrained_model_name)
	path_best_model = os.path.join(save_model_path, args.best_model_folder)

    # initial the network
	My_XLMRobertaModel = My_XLMRobertaModel(pretrain_model_name = args.pretrained_model_name,
		                                    path_saved_model = path_saved,
		                                    learning_rate = args.train_lr,
		                                    n_class = args.num_class,
		                                    epochs = args.num_train_epoch,
											warm_step=args.num_warm_step,
		                                    train_batch_size = args.batch_size_tra,
		                                    test_batch_size = args.batch_size_tes,
		                                    max_len = args.token_max_len,
		                                    gpu = args.if_gpu
											)
	print('[step-1] The model is initialized')

	processor = ProcessData(tokenizer_vocab_path = path_best_model,
							tokenizer_max_len = args.token_max_len,
							feature_name=args.data_feature_type #choose traslated text or original text 
							)
	print('[step-2] The data processor is initialized')

	# load the trained model
	model_loaded = My_XLMRobertaModel.load_model(path_best_model)
	print('[step-3] The best performance model is loaded')

	# preprocess the new data
	test,y_true = processor.prepare_data_loader(path_test_data,
												path_save_split= args.path_project_folder,
	                                            mode_test=args.data_prepare_mode
												)											
	print('[step-4] The preprocessing and dataloader on new data is finished')
	
	# make prediction on new data											
	preds = My_XLMRobertaModel.eval_model(model_loaded,test,y_true)
	print('[step-5] The predicted label is generated')

	# save prediction into csv file
	df=pd.DataFrame(preds,columns=['predicted_label'])
	df.to_csv(os.path.join(args.path_project_folder,'predicted_label_on_test.csv'), index=False)
	print('The whole testing procedure spends %.3f minuts'%((time.time()-start_time)/60))