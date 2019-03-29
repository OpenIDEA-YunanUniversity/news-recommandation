import pickle
import os
import pandas as pd
import numpy as np


def load_article_embending():
	artiles = open('../data/articles_embeddings.pickle','rb')
	data = pickle.load(artiles)

	return data


def load_clicks():
	if os.path.exists('../data/click.csv'):
		data = pd.read_csv('../data/click.csv')
		return data
	root_click_folder = '../data/click/'
	click_files = os.listdir(root_click_folder)
	click_df = pd.DataFrame()
	for click_file in click_files:
		try:
			abs_click_path = root_click_folder+click_file
			print(abs_click_path)
			current_click_df = pd.read_csv(abs_click_path)
			click_df = click_df.append(current_click_df,ignore_index=True)
		except:
			continue
	click_df.to_csv('../data/click.csv',index=False)

def generate_training_data():
	if os.path.exists('../data/all_clicks.csv'):
		all_clicks = pd.read_csv('../data/all_clicks.csv')
		return all_clicks
	clicks = load_clicks()
	clicks = clicks[:20000]
	articles_embeddings = load_article_embending()
	unique_user_vec = np.unique(clicks['user_id'])
	
	print('user num is {}'.format(unique_user_vec.shape[0]))
	user_didnt_read_info = []
	count=1;
	user_nums = int(0.2*unique_user_vec.shape[0])
	
	for user in unique_user_vec[:int(0.001*unique_user_vec.shape[0])]:
		print('calculating user {}, rest number of {} user to calculate'.format(user,user_nums-count))
		user_have_read_articles = np.unique(clicks[clicks['user_id']==user]['click_article_id'])
		user_have_read_articles_nums = user_have_read_articles.shape[0]
		user_didnt_read_articles = clicks[~clicks['click_article_id'].isin(user_have_read_articles)]#[:int(0.4*user_have_read_articles_nums)]
		user_didnt_read_info.extend(user_didnt_read_articles.values.tolist())
		count+=1

	user_didnt_read_info = pd.DataFrame(user_didnt_read_info,columns=clicks.columns.tolist())
	user_didnt_read_info = user_didnt_read_info.drop_duplicates(subset=['user_id','click_article_id'])
	user_didnt_read_info['label']=0
	user_didnt_read_info = user_didnt_read_info[:int(0.3*clicks.shape[0])]
	print(user_didnt_read_info)
	
	clicks['label'] = 1
	all_clicks = clicks.append(user_didnt_read_info, ignore_index=True)
	all_clicks = all_clicks[['user_id','click_article_id','label']]
	articles_embeddings = pd.DataFrame(articles_embeddings)
	articles_embeddings['click_article_id'] = articles_embeddings.index
	all_clicks = all_clicks.merge(articles_embeddings,how='left',on='click_article_id')
	all_clicks.to_csv('../data/all_clicks.csv',index=False)
	#print(all_clicks.isna().any())
	#print(all_clicks[:10])

if __name__ == '__main__':
	generate_training_data()