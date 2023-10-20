import os,random,time,argparse,json
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AdamW
from transformers import XLMRobertaTokenizer,XLMRobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score
from sklearn.metrics import classification_report
import torch
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")# remove UndefinedMetricWarning: Precision is ill-defined


class preprocess:
	def __init__(self,tokenizer_vocab_path,tokenizer_max_len):
		self.tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_vocab_path) # load the pretained model
		self.max_len = tokenizer_max_len

	def encode_fn(self,text_single):
		'''
		Using tokenizer to preprocess the text 
		example of text_single:'today is a good daya'
		'''
		tokenizer = self.tokenizer(text_single,
									padding = True,
									truncation = True,
									max_length = self.max_len,
									return_tensors='pt'
									)
		input_ids = tokenizer['input_ids']
		attention_mask = tokenizer['attention_mask']
		return input_ids,attention_mask

	def process_tokenizer(self, text_single):
		'''
		Preprocess text and prepare dataloader for a single new sentence
		'''
		input_ids,attention_mask = self.encode_fn(text_single)
		data = TensorDataset(input_ids,attention_mask)
		return data


class My_XLMRobertaModel:
	def __init__(self,pretrain_model_path,
	                  path_saved_model,
					  learning_rate,
					  n_class,
					  path_table,
					  epochs,
					  warm_step,
					  train_batch_size,
					  test_batch_size,
					  max_len,
					  gpu=True):
		self.n_class = n_class # number of class for classification
		self.path_table = pd.read_csv(path_table)
		self.max_len = max_len # max sentence length
		self.lr = learning_rate # learning rate
		self.epochs = epochs
		self.warm_step=warm_step
		
		self.batch_size = train_batch_size # batch_size for model training
		self.val_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.softmax = torch.nn.Softmax(dim=1)
			
		self.path_saved_model = path_saved_model # path to save model checkpoints
		self.gpu = gpu # if using GPU for model training
		self.model = XLMRobertaForSequenceClassification.from_pretrained(pretrain_model_path)
		self.model.eval()# set pytorch model for inference mode

		# using parallel strategy if having multiple GPU
		if torch.cuda.device_count()>1:
			self.model = torch.nn.DataParallel(self.model)

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

		self.model = self.model.to(self.device)

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

		    # Evaluate model on validation set after each training epoch
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
		Save fine-tuned BERT model
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
		
		preds = []
		for i, batch in enumerate(tqdm(pred_dataloader)):
			with torch.no_grad():
				outputs = model(input_ids=batch[0].to(self.device),
								attention_mask=batch[1].to(self.device)
								)
				loss, logits = outputs.loss, outputs.logits
				logits = logits.detach().cpu().numpy()
				preds += list(np.argmax(logits, axis=1))
		return preds
	
	def predict_single(self, model, pred_data):
		''' 
		Model inference for new single sentence
		'''
		pred_dataloader = DataLoader(pred_data, batch_size=self.test_batch_size, shuffle=False)
		
		for i, batch in enumerate(pred_dataloader):
			with torch.no_grad():
				outputs = model(input_ids=batch[0].to(self.device),
								attention_mask=batch[1].to(self.device)
								)
				loss, logits = outputs.loss, outputs.logits
				probability = self.softmax(logits)
				probability_list = probability.detach().cpu().numpy()
				
		return probability_list

	def transform_output(self,pred):
		'''
		transform the model output into a dictonary with  all intents and its probability
		'''
		final={}

		# transform the relation table(between label and intent)
		dict_label_intent= self.path_table[["intent","corresponding_label"]].set_index("corresponding_label").to_dict()['intent']
		
		# transform the output into dictionary(between intent and probability)
		for idx in range(pred.shape[1]):
			final[dict_label_intent[idx]]=pred[0][idx]
		
		return final
	
	def inference(self,prepared_data):
		'''
		make prediction on one new sentence and output a json format variable
		'''
		tmp=[]
		prob_distribution = self.predict_single(self.model, prepared_data)
		result = self.transform_output(prob_distribution.astype(np.float))
		tmp.append(result)
		final = json.dumps(tmp)

		return final

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path_project_folder', default=r"/home/TRIM-AI/")
	parser.add_argument('--new_sentence', default=r"Hello,how are mtoto wa umri gani?")
	parser.add_argument('--name_table', default="intent_label_relation.csv")# csv for relationship between numeric label and true intent
	parser.add_argument('--pretrained_model_name', default='xlm-roberta-large')
	parser.add_argument('--best_model_folder', default='xlm-roberta-large-2')
	parser.add_argument('--num_train_epoch', type=int, default=3)
	parser.add_argument('--num_warm_step', type=int, default=100)
	parser.add_argument('--train_lr', type=int, default=2e-5)
	parser.add_argument('--num_class', type=int, default=58) # number of classes for text classification
	parser.add_argument('--batch_size_tra', type=int, default=8) # training and validation batch size
	parser.add_argument('--batch_size_tes', type=int, default=128) # inference batch size
	parser.add_argument('--token_max_len', type=int, default=250) # max_len that tokenizer will process
	parser.add_argument('--if_gpu', default=True) # max_len that tokenizer will process
	args = parser.parse_args()
	
    # set the dataset path
	path_table = os.path.join(args.path_project_folder, args.name_table)

	# set trained model path
	save_model_path = os.path.join(args.path_project_folder, "checkpoint")
	path_saved = os.path.join(save_model_path, args.pretrained_model_name)
	path_best_model = os.path.join(save_model_path, args.best_model_folder)

    # step-1 load the model
	My_XLMRobertaModel = My_XLMRobertaModel(pretrain_model_path = path_best_model,
		                                    path_saved_model = path_saved,
		                                    learning_rate = args.train_lr,
		                                    n_class = args.num_class,
											path_table=path_table,
		                                    epochs = args.num_train_epoch,
											warm_step=args.num_warm_step,
		                                    train_batch_size = args.batch_size_tra,
		                                    test_batch_size = args.batch_size_tes,
		                                    max_len = args.token_max_len,
		                                    gpu = args.if_gpu
											)

	processor = preprocess(tokenizer_vocab_path = path_best_model,
							tokenizer_max_len = args.token_max_len
							)
	print('[step-1] The model and data processor are initialized')
	
	# step-2 preprocess the new single sentence
	text= processor.process_tokenizer(text_single=args.new_sentence)											
	print('[step-2] The preprocessing step is finished')
	
	# step-3 make prediction on new single sentence											
	preds = My_XLMRobertaModel.inference(text)
	print('[step-3] The predicted label is generated')
	print('[Total]The whole procedure is finished')
	print(preds)